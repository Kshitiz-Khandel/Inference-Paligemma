## Inference.py for Paligemma (Modified for Batch Inference)
## Flash Attention

from PIL import Image
import torch
import fire
from typing import List, Optional, Tuple

from processing_paligemma import PaliGemmaProcessor
from gemma_flash import KVCache, PaliGemmaForConditionalGeneration
#from gemma_decoder import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompts: List[str], image_file_paths: List[str], device: str
):
    images = []
    for f_path in image_file_paths:
        try:
            img = Image.open(f_path).convert("RGB")
            images.append(img)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {f_path}. Skipping this image.")
        except Exception as e:
            print(f"Warning: Failed to open image {f_path}: {e}. Skipping this image.")

    if not images:
        raise ValueError("No valid images found for processing.")
    if len(images) != len(prompts):
        raise ValueError(
            f"Mismatch: Successfully loaded {len(images)} images, but have {len(prompts)} prompts. "
            "Ensure all image files exist and are accessible."
        )

    model_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def _sample_top_p(probs: torch.Tensor, p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    sum_probs_sort = probs_sort.sum(dim=-1, keepdim=True)
    probs_sort = torch.where(
        sum_probs_sort == 0,
        torch.ones_like(probs_sort) / probs_sort.shape[-1],
        probs_sort,
    )
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token










def test_batch_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompts: List[str],
    image_file_paths: List[str],
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    # --- Prefill Step ---
    model_inputs = get_model_inputs(processor, prompts, image_file_paths, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"] # Get pixel values once
    print("DEBUG shapes: input_ids", input_ids.shape, "attention_mask", attention_mask.shape, "pixel_values", pixel_values.shape)

    batch_size = input_ids.shape[0]
    
    # ✅ Initialize kv_cache to None, it will be created on the first model call
    kv_cache = None
    generated_tokens_batch = [[] for _ in range(batch_size)]
    active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

    # --- Profiler Setup ---
    tb_logdir = "/home/jupyter/Paligemma2/PyTorch-PaliGemma-2/logs/tb_logdir_batch/output"
    prof_sched = schedule(wait=1, warmup=2, active=5, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        on_trace_ready=tensorboard_trace_handler(tb_logdir),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        
        # --- Decoding Loop ---
        for i in range(max_tokens_to_generate):
            if not active_sequences.any():
                break

            # ✅ KEY CHANGE: Pass pixel_values only for the first token (prefill)
            current_pixel_values = pixel_values if i == 0 else None

            outputs = model(
                input_ids=input_ids,
                pixel_values=current_pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            
            # ✅ CORRECTLY UPDATE a single kv_cache object
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]

            # --- (Rest of your sampling and token management logic is mostly correct) ---
            
            # Filter logits for active sequences
            active_logits = next_token_logits[active_sequences]

            if do_sample:
                next_tokens_active = _sample_top_p(active_logits / temperature, top_p)
            else:
                next_tokens_active = torch.argmax(active_logits, dim=-1, keepdim=True)
            
            # Create a full batch tensor for the next tokens, filling inactive with pad
            next_tokens = torch.full(
                (batch_size, 1),
                fill_value=processor.tokenizer.pad_token_id,
                dtype=torch.long,
                device=device
            )
            next_tokens[active_sequences] = next_tokens_active

            # Append generated tokens
            for j in range(batch_size):
                if active_sequences[j]:
                    generated_tokens_batch[j].append(next_tokens[j])

            # Update active sequences
            newly_stopped = (next_tokens.squeeze(-1) == processor.tokenizer.eos_token_id)
            active_sequences = active_sequences & (~newly_stopped)

            # ✅ Set up inputs for the NEXT iteration
            input_ids = next_tokens
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((batch_size, 1), dtype=torch.long, device=device)
            ], dim=-1)

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # --- (Decoding and printing logic remains the same) ---
    final_decoded_outputs = []
    for i in range(batch_size):
        if generated_tokens_batch[i]:
            decoded = processor.tokenizer.decode(
                torch.cat(generated_tokens_batch[i]), skip_special_tokens=True
            )
        else:
            decoded = "(No tokens generated)"
        final_decoded_outputs.append(f"{prompts[i]}: {decoded.strip()}")

    print("\n-----------------------------------------------------------------------")
    for i, output in enumerate(final_decoded_outputs):
        print(f"Output {i+1}:\n{output}")
        print("-" * 30)
    print("-----------------------------------------------------------------------\n")


def main(
    model_path: str = None,
    prompts: str = None,
    image_file_paths: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    print("Device in use: ", device)

    print(f"Loading model from {model_path}")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    if device == "cuda":
        model = model.to(torch.bfloat16)

    processor = PaliGemmaProcessor(tokenizer, model.config.text_config.num_image_tokens, model.config.vision_config.image_size)

    prompts_list = [p.strip() for p in prompts.split(',') if p.strip()] if prompts else []
    image_file_paths_list = [f.strip() for f in image_file_paths.split(',') if f.strip()] if image_file_paths else []

    if len(prompts_list) != len(image_file_paths_list):
        raise ValueError("Number of prompts must match number of image file paths for batch inference.")
    if not prompts_list:
        print("No prompts or image paths provided. Exiting.")
        return

    print(f"Running batch inference for {len(prompts_list)} samples")
    with torch.no_grad():
        test_batch_inference(
            model,
            processor,
            device,
            prompts_list,
            image_file_paths_list,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)