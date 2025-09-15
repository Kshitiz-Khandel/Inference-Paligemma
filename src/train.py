import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from datasets import load_dataset, VerificationMode
from transformers import AutoProcessor
from PIL import Image, UnidentifiedImageError
import requests
from utils import load_hf_model
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# ========================
# CONFIG
# ========================
TARGET_PATH = "/home/jupyter/Paligemma/google/paligemma-3b-pt-896"
DRAFT_PATH  = "/home/jupyter/Paligemma/google/draft/draft_distilled"
SAVE_PATH   = "/home/jupyter/Paligemma/google/draft_distilled_trained"

import os
from io import BytesIO
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from datasets import load_dataset, VerificationMode
from transformers import AutoProcessor
from PIL import Image, ImageFile, UnidentifiedImageError
import requests
from utils import load_hf_model
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========================
# CONFIG


BATCH_SIZE = 4  # smaller batch to avoid OOM
LR = 1e-5
EPOCHS = 10
MAX_LEN = 128
TEMPERATURE = 2.0
ALPHA = 0.5

# ========================
# DATASET & COLLATE
# ========================
class ImageCaptionDataset(Dataset):
    def __init__(self, split="train[:30000]"):
        print("Loading dataset...")
        self.dataset = load_dataset(
            "conceptual_captions",
            split=split,
            verification_mode=VerificationMode.NO_CHECKS
        )
        print(f"Dataset loaded with {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def collate_fn_wrapper():
    processor = AutoProcessor.from_pretrained(TARGET_PATH, padding_side='right')

    def is_valid_pil(img):
        return isinstance(img, Image.Image)

    def safe_open_image_from_url(url, timeout=10):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            content = resp.content
            img = Image.open(BytesIO(content))
            if not is_valid_pil(img):
                return None
            img = img.convert("RGB")
            if img.width <= 0 or img.height <= 0:
                return None
            return img
        except Exception as e:
            print(f"Bad image {url}: {e}")
            return None

    def collate_fn(batch):
        images, captions = [], []
        for ex in batch:
            url, caption = ex.get("image_url"), ex.get("caption")
            if not url or not caption:
                continue
            img = safe_open_image_from_url(url)
            if img is None:
                continue
            images.append(img)
            captions.append("<image>\n" + caption)

        if not images:
            return None

        try:
            return processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN
            )
        except Exception as e:
            return None

    return collate_fn


# ========================
# DISTILLATION LOSS
# ========================
def distillation_loss(student_logits, teacher_logits, labels, temperature=TEMPERATURE, alpha=ALPHA):
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1)
    )
    log_p = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    q = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = torch.nn.functional.kl_div(log_p, q.detach(), reduction="batchmean") * (temperature ** 2)
    return alpha * ce_loss + (1 - alpha) * kl_loss


# ========================
# DDP SETUP
# ========================
def setup_ddp():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# ========================
# MAIN
# ========================
def main():
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # --- Load models ---
    teacher = None
    if rank == 0:
        print("Loading teacher model on GPU (bfloat16)...")
        teacher, _ = load_hf_model(TARGET_PATH, "cuda")
        teacher.to(torch.bfloat16).eval()
        for p in teacher.parameters():
            p.requires_grad = False

    print(f"Rank {rank}: Loading student model on GPU...")
    student, _ = load_hf_model(DRAFT_PATH, "cuda")
    student.train()
    student = DDP(student, device_ids=[local_rank])

    # --- Dataset ---
    dataset = ImageCaptionDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_fn_wrapper(),
        num_workers=4
    )

    optimizer = AdamW(student.parameters(), lr=LR)
    scaler = GradScaler(enabled=True)

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
        for batch in progress_bar:
            if batch is None:
                continue

            # --- Teacher logits only on rank 0 ---
            if rank == 0:
                gpu_batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher_logits = teacher(**gpu_batch)['logits']
            else:
                teacher_logits = None

            # --- Broadcast teacher logits to all ranks in FP16 ---
            teacher_logits = dist.broadcast_object_list([teacher_logits], src=0)[0]

            # --- Student forward ---
            gpu_batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = student(**gpu_batch)['logits']
                loss = distillation_loss(student_logits, teacher_logits.to(torch.bfloat16), gpu_batch["input_ids"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                progress_bar.set_postfix({"loss": loss.item()})

    # --- Save ---
    if rank == 0:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(student.module.state_dict(), f"{SAVE_PATH}.pt")
        print(f"✅ Distilled student saved at {SAVE_PATH}.pt")

    dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    main()
