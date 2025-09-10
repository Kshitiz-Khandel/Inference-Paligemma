# # train_ddp.py
# import os
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.distributed import DistributedSampler
# from torch.optim import AdamW
# from datasets import load_dataset
# from transformers import AutoProcessor
# from PIL import Image, UnidentifiedImageError
# import requests
# from utils import load_hf_model
# from torch.cuda.amp import autocast, GradScaler
# from tqdm import tqdm

# # ========================
# # CONFIG
# # ========================
# TARGET_PATH = "/home/jupyter/Paligemma/google/paligemma-3b-pt-896"
# DRAFT_PATH  = "/home/jupyter/Paligemma/google/draft/"
# SAVE_PATH   = "/home/jupyter/Paligemma/google/draft_distilled_ddp"

# BATCH_SIZE = 8       # Reduced for stability, can increase if memory allows
# LR = 1e-5
# EPOCHS = 1
# MAX_LEN = 128
# TEMPERATURE = 2.0
# ALPHA = 0.5

# # ========================
# # DATASET & COLLATE
# # ========================
# class ImageCaptionDataset(Dataset):
#     def __init__(self, split="train[:1%]"):
#         print("Loading dataset...")
#         # Using a more standard and reliable dataset for this example
#         full_dataset = load_dataset("conceptual_captions", split=split)
#         self.dataset = full_dataset
#         print(f"Dataset loaded with {len(self.dataset)} samples.")

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

# def collate_fn_wrapper():
#     """Wrapper to initialize processor once."""
#     processor = AutoProcessor.from_pretrained(TARGET_PATH)
    
#     def collate_fn(batch):
#         images, captions = [], []
#         for ex in batch:
#             url, caption = ex["image_url"], ex["caption"]
#             try:
#                 # Use a session for potential connection reuse
#                 with requests.Session() as s:
#                     img = Image.open(s.get(url, stream=True, timeout=10).raw).convert("RGB")
#                 images.append(img)
#                 captions.append(caption)
#             except (UnidentifiedImageError, requests.RequestException, OSError) as e:
#                 # print(f"Skipping image due to error: {e}")
#                 continue

#         if not images:
#             return None

#         try:
#             enc = processor(
#                 images=images,
#                 text=captions,
#                 return_tensors="pt",
#                 padding="max_length", # Use max_length for consistent tensor shapes
#                 truncation=True,
#                 max_length=MAX_LEN
#             )
#             return enc
#         except Exception as e:
#             print(f"Processor failed with error: {e}")
#             return None
#     return collate_fn

# # ========================
# # DISTILLATION LOSS
# # ========================
# def distillation_loss(student_logits, teacher_logits, labels, temperature=TEMPERATURE, alpha=ALPHA):
#     ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
#         student_logits.view(-1, student_logits.size(-1)),
#         labels.view(-1)
#     )
    
#     log_p = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
#     q = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    
#     kl_loss = torch.nn.functional.kl_div(log_p, q.detach(), reduction="batchmean") * (temperature ** 2)
    
#     return alpha * ce_loss + (1 - alpha) * kl_loss

# # ========================
# # DDP SETUP
# # ========================
# def setup_ddp():
#     dist.init_process_group("nccl")
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     torch.cuda.set_device(rank)
#     return rank, world_size

# def cleanup_ddp():
#     dist.destroy_process_group()

# # ========================
# # MAIN TRAIN FUNCTION
# # ========================
# def main():
#     rank, world_size = setup_ddp()
#     device = torch.device(f"cuda:{rank}")

#     # --- Load Models ---
#     # Teacher is loaded ONLY on rank 0 and stays on CPU to save GPU memory
#     teacher = None
#     if rank == 0:
#         print("Loading teacher model on CPU...")
#         # Load model on CPU first, then optionally change dtype
#         teacher, _ = load_hf_model(TARGET_PATH, "cpu")
#         teacher.to(torch.bfloat16).eval() # Use bfloat16 for faster CPU inference
#         for p in teacher.parameters():
#             p.requires_grad = False
    
#     print(f"Rank {rank}: Loading student model on GPU...")
#     student, _ = load_hf_model(DRAFT_PATH, device)
#     student.train()
#     student = DDP(student, device_ids=[rank], find_unused_parameters=False)

#     # --- Dataset & DataLoader ---
#     dataset = ImageCaptionDataset()
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn_wrapper(), num_workers=4, pin_memory=True)

#     optimizer = AdamW(student.parameters(), lr=LR)
#     scaler = GradScaler()

#     for epoch in range(EPOCHS):
#         sampler.set_epoch(epoch)
#         progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
        
#         for step, batch in enumerate(progress_bar):
#             if batch is None:
#                 continue

#             # Move batch to the current device
#             batch = {k: v.to(device) for k, v in batch.items()}
            
#             # --- Teacher Forward Pass (on Rank 0) and Broadcast ---
#             if rank == 0:
#                 with torch.no_grad(), autocast(device_type='cpu', dtype=torch.bfloat16):
#                     teacher_logits = teacher(**batch).logits
#                 # Ensure logits are on the correct device for broadcasting
#                 teacher_logits = teacher_logits.to(device)
#             else:
#                 # Create a placeholder tensor on other ranks
#                 vocab_size = student.module.config.text_config.vocab_size
#                 teacher_logits = torch.empty((batch['input_ids'].size(0), MAX_LEN, vocab_size), device=device, dtype=torch.float32)

#             dist.broadcast(teacher_logits, src=0)

#             # --- Student Forward and Backward Pass ---
#             optimizer.zero_grad(set_to_none=True)
#             with autocast(dtype=torch.bfloat16):
#                 student_logits = student(**batch).logits
#                 loss = distillation_loss(student_logits, teacher_logits.to(torch.bfloat16), batch["input_ids"])

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             if rank == 0 and step % 10 == 0:
#                 progress_bar.set_postfix({"loss": loss.item()})

#     # --- Save Model ---
#     if rank == 0:
#         os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
#         torch.save(student.module.state_dict(), f"{SAVE_PATH}.pt")
#         print(f"✅ Distilled student saved at {SAVE_PATH}.pt")

#     cleanup_ddp()

# if __name__ == "__main__":
#     main()









# train_ddp.py
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
DRAFT_PATH  = "/home/jupyter/Paligemma/google/draft/"
SAVE_PATH   = "/home/jupyter/Paligemma/google/draft_distilled_ddp"

BATCH_SIZE = 8       # Per-GPU batch size. Start small.
LR = 1e-5
EPOCHS = 1
MAX_LEN = 128
TEMPERATURE = 2.0
ALPHA = 0.5

# ========================
# DATASET & COLLATE
# ========================
class ImageCaptionDataset(Dataset):
    def __init__(self, split="train[:100]"):
        print("Loading dataset...")
        # This dataset is generally more reliable for downloading images.
        self.dataset = load_dataset("conceptual_captions", split=split, verification_mode=VerificationMode.NO_CHECKS)
        print(f"Dataset loaded with {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_fn_wrapper():
    """Wrapper to initialize processor once per worker."""
    processor = AutoProcessor.from_pretrained(TARGET_PATH, padding_side='right')
    
    def collate_fn(batch):
        images, captions = [], []
        for ex in batch:
            url, caption = ex["image_url"], ex["caption"]
            try:
                img = Image.open(requests.get(url, stream=True, timeout=10).raw).convert("RGB")
                images.append(img)
                # Manually add the <image> token as recommended
                captions.append("<image>\n" + caption)
            except (UnidentifiedImageError, requests.RequestException, OSError):
                continue

        if not images: return None

        try:
            enc = processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN
            )
            return enc
        except Exception:
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
# DDP SETUP & MAIN
# ========================
def setup_ddp():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    # --- Load Models ---
    teacher = None
    if rank == 0:
        print("Loading teacher model on CPU...")
        teacher, _ = load_hf_model(TARGET_PATH, "cpu")
        teacher.to(torch.bfloat16).eval()
        for p in teacher.parameters(): p.requires_grad = False
    
    print(f"Rank {rank}: Loading student model on {device}...")
    student, _ = load_hf_model(DRAFT_PATH, "cpu") # Load to CPU first
    student = student.to(device) # Then move to GPU
    student.train()
    student = DDP(student, device_ids=[int(os.environ["LOCAL_RANK"])])

    # --- Dataset & DataLoader ---
    dataset = ImageCaptionDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn_wrapper(), num_workers=4)

    optimizer = AdamW(student.parameters(), lr=LR)
    scaler = GradScaler(enabled=True)

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
        
        for batch in progress_bar:
            if batch is None: continue
            
            # --- Teacher Forward (CPU on rank 0) & Broadcast ---
            if rank == 0:
                cpu_batch = {k: v.to("cpu") for k, v in batch.items()}
                with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                    teacher_logits = teacher(**cpu_batch).logits.to(device) # Move result to GPU
            else:
                vocab_size = student.module.config.text_config.vocab_size
                teacher_logits = torch.empty((batch['input_ids'].size(0), MAX_LEN, vocab_size), device=device, dtype=torch.float32)

            dist.broadcast(teacher_logits, src=0)
            
            # --- Student Forward & Backward (GPU on all ranks) ---
            gpu_batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                student_logits = student(**gpu_batch).logits
                loss = distillation_loss(student_logits, teacher_logits.to(torch.bfloat16), gpu_batch["input_ids"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                progress_bar.set_postfix({"loss": loss.item()})

    # --- Save Model ---
    if rank == 0:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        # Unwrap the model from DDP before saving
        torch.save(student.module.state_dict(), f"{SAVE_PATH}.pt")
        print(f"✅ Distilled student saved at {SAVE_PATH}.pt")

    cleanup_ddp()

if __name__ == "__main__":
    main()