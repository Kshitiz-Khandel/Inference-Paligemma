## Inference.py for Paligemma (Modified for Batch Inference)

from PIL import Image
import torch
import fire
from typing import List, Optional, Tuple

from processing_paligemma import PaliGemmaProcessor
from gemma_decoder import KVCache, PaliGemmaForConditionalGeneration
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
            img = Image.open(f_path).convert("RGB") # Ensure consistent mode
            images.append(img)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {f_path}. Skipping this image.")
        except Exception as e:
            print(f"Warning: Failed to open image {f_path}: {e}. Skipping this image.")

    if not images:
        raise ValueError("No valid images found for processing.")
    if len(images) != len(prompts):
        # This can happen if some image files were not found
        raise ValueError(
            f"Mismatch: Successfully loaded {len(images)} images, but have {len(prompts)} prompts. "
            "Ensure all image files exist and are accessible."
        )

    model_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def _sample_top_p(probs: torch.Tensor, p: float):
    # probs: (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    # Handle cases where all probabilities become zero after masking
    sum_probs_sort = probs_sort.sum(dim=-1, keepdim=True)
    # If a row sums to 0, distribute probability uniformly to avoid NaNs.
    # This ensures multinomial doesn't fail.
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
    model_inputs = get_model_inputs(processor, prompts, image_file_paths, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    batch_size = input_ids.shape[0]
    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens_batch: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
    
    # Track which sequences are still active (haven't generated EOS)
    active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Profiler setup (as in your reference script)
    tb_logdir = "/home/jupyter/Paligemma2/PyTorch-PaliGemma-2/logs/tb_logdir_batch"
    prof_sched = schedule(
        wait=1,
        warmup=2,
        active=5,
        repeat=1
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        on_trace_ready=tensorboard_trace_handler(tb_logdir),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        for i in range(max_tokens_to_generate):
            # Break if all sequences have stopped
            if not active_sequences.any():
                break

            # Only consider active sequences for the current forward pass
            # We need to filter inputs for the current active sequences only
            current_input_ids = input_ids[active_sequences]
            current_attention_mask = attention_mask[active_sequences]
            current_pixel_values=pixel_values
            #current_pixel_values = pixel_values[active_sequences] if i == 0 else None # Vision model only processes once

            # When processing in batches, ensure the KV cache is updated for the full batch,
            # even if some sequences are inactive. The model's internal attention mechanism
            # (especially with padding) will handle inactive sequences appropriately.
            # The kv_cache stores for all sequences initially.
            
            # The model's forward pass expects full batch, so we need to pass full input_ids and attention_mask
            # and then handle the logits and next_token based on active_sequences.
            # The KV cache update should also happen for all sequences.

            outputs = model(
                input_ids=input_ids, # Pass full input_ids (will contain generated tokens for active, EOS for inactive)
                #pixel_values=pixel_values if i == 0 else None, # Pass original pixel_values only for the first step
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :] # Logits for the last token of each sequence in the batch

            # Filter logits for active sequences
            next_token_logits_active = next_token_logits[active_sequences]

            if do_sample:
                next_token_active = _sample_top_p(next_token_logits_active / temperature, top_p)
            else:
                next_token_active = torch.argmax(next_token_logits_active, dim=-1, keepdim=True)
            
            # Create a full batch-sized tensor for the next tokens
            # Initialize with padding_token_id or stop_token for inactive sequences
            next_token_full_batch = torch.full(
                (batch_size, 1),
                fill_value=processor.tokenizer.pad_token_id, # Use pad_token_id for inactive sequences to not influence generation
                dtype=torch.long,
                device=device
            )
            next_token_full_batch[active_sequences] = next_token_active.squeeze(-1).unsqueeze(-1) # Ensure shape (N_active, 1)

            # Append generated tokens for active sequences only
            # The generated_tokens_batch should accumulate tokens per original batch item
            current_next_tokens_list = next_token_full_batch.squeeze(-1).tolist()
            for j in range(batch_size):
                if active_sequences[j]:
                    generated_tokens_batch[j].append(torch.tensor([current_next_tokens_list[j]], device=device))

            # Update active sequences based on stop token
            # If a token is stop_token, that sequence becomes inactive.
            newly_stopped = (next_token_full_batch.squeeze(-1) == stop_token)
            active_sequences = active_sequences & (~newly_stopped)

            # Update input_ids for the next iteration: it's now just the newly generated tokens
            input_ids = next_token_full_batch

            # Update attention_mask for the next iteration (append a 1 for the new token for all original sequences)
            # The attention mask needs to grow for all sequences in the batch, padding appropriately.
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=input_ids.device)], dim=-1
            )
            
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    final_decoded_outputs = []
    for i in range(batch_size):
        if generated_tokens_batch[i]:
            decoded = processor.tokenizer.decode(
                torch.cat(generated_tokens_batch[i], dim=-1), skip_special_tokens=True
            )
        else:
            decoded = "(No tokens generated)" # Fallback if for some reason no tokens were generated

        final_decoded_outputs.append(f"{prompts[i]}: {decoded.strip()}")

    print("\n-----------------------------------------------------------------------")
    for i, output in enumerate(final_decoded_outputs):
        print(f"Output {i+1}:\n{output}")
        print("-" * 30)
    print("-----------------------------------------------------------------------\n")


def main(
    model_path: str = None,
    prompts: str = None,  # Expecting a comma-separated string for multiple prompts
    image_file_paths: str = None,  # Expecting a comma-separated string for multiple image paths
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
        # Using bfloat16 and torch.compile for performance on CUDA-enabled GPUs
        model = model.to(torch.bfloat16)
        # apply_fake_sparsity(model) # This prunes weights to 2:4 pattern (makes values zero)
        # apply_sparse_semistructured(model)
        
        model = torch.compile(model)

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Convert comma-separated strings to lists
    prompts_list = [p.strip() for p in prompts.split(',') if p.strip()] if prompts else []
    image_file_paths_list = [f.strip() for f in image_file_paths.split(',') if f.strip()] if image_file_paths else []

    # Ensure number of prompts matches number of images
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
    torch.cuda.empty_cache() # Clear CUDA cache at start
    fire.Fire(main)













