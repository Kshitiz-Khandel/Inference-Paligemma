## inference with speculative decoding
from PIL import Image
import torch
import torch.nn.functional as F
import fire
from typing import List, Optional, Tuple
import time

from processing_paligemma import PaliGemmaProcessor
from gemma_flash import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    """Move model inputs to the specified device."""
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, 
    prompts: List[str], 
    image_file_paths: List[str], 
    device: str
):
    """Load images and prepare model inputs."""
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


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    """Convert logits to probabilities with optional top-k filtering."""
    print(f"[LOGITS_TO_PROBS] Input logits shape: {logits.shape}, temperature: {temperature}, top_k: {top_k}")
    
    logits = logits / max(temperature, 1e-5)
    
    if top_k is not None:
        print(f"[LOGITS_TO_PROBS] Applying top-k filtering with k={top_k}")
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print(f"[LOGITS_TO_PROBS] Output probs shape: {probs.shape}")
    return probs


def _sample_top_p(logits, top_p=0.9):
    """Ultra-robust top-p sampling with extensive error handling."""
    print(f"[TOP_P_SAMPLING] Input logits shape: {logits.shape}, top_p: {top_p}")
    
    # Handle edge cases
    if logits.numel() == 0:
        raise ValueError("Empty logits tensor")
    
    # Step 1: Handle extreme logits values
    logits_min, logits_max = logits.min().item(), logits.max().item()
    print(f"[TOP_P_SAMPLING] Input logits range: [{logits_min:.3f}, {logits_max:.3f}]")
    
    if not torch.isfinite(logits).all():
        print(f"[TOP_P_SAMPLING] WARNING: Non-finite logits detected, clamping")
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    
    # Clamp to reasonable range
    logits = torch.clamp(logits, min=-100.0, max=100.0)
    
    # Step 2: Compute probabilities with numerical stability
    # Subtract max for numerical stability
    logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
    
    # Compute probabilities
    try:
        probs = F.softmax(logits_shifted, dim=-1)
    except Exception as e:
        print(f"[TOP_P_SAMPLING] Softmax failed: {e}, using uniform distribution")
        probs = torch.ones_like(logits) / logits.shape[-1]
    
    # Step 3: Check for invalid probabilities
    if not torch.isfinite(probs).all():
        print(f"[TOP_P_SAMPLING] Non-finite probabilities detected, fixing")
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Ensure probabilities are positive
    probs = torch.clamp(probs, min=1e-12)
    
    # Renormalize to ensure sum = 1
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    print(f"[TOP_P_SAMPLING] Probs range: [{probs.min().item():.6f}, {probs.max().item():.6f}]")
    print(f"[TOP_P_SAMPLING] Probs sum: {probs.sum(dim=-1).item():.6f}")
    
    # Step 4: Apply top-p filtering
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask: keep tokens where cumsum - current_prob <= top_p
    # This means we keep tokens that are needed to reach the top_p threshold
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # Always keep the most probable token
    
    # Set probabilities of removed tokens to 0
    sorted_probs_filtered = sorted_probs.clone()
    sorted_probs_filtered[sorted_indices_to_remove] = 0.0
    
    # Step 5: Final probability validation and normalization
    total_prob = sorted_probs_filtered.sum(dim=-1, keepdim=True)
    
    if total_prob.item() < 1e-10:
        print(f"[TOP_P_SAMPLING] All probabilities filtered out, using greedy")
        selected_token = sorted_indices[..., 0:1]
    else:
        # Renormalize filtered probabilities
        sorted_probs_filtered = sorted_probs_filtered / total_prob
        
        # Final validation - check for any remaining issues
        has_nan = torch.isnan(sorted_probs_filtered).any()
        has_inf = torch.isinf(sorted_probs_filtered).any()
        has_negative = (sorted_probs_filtered < 0).any()
        
        if has_nan or has_inf or has_negative:
            print(f"[TOP_P_SAMPLING] Invalid filtered probabilities detected:")
            print(f"[TOP_P_SAMPLING]   NaN: {has_nan}, Inf: {has_inf}, Negative: {has_negative}")
            print(f"[TOP_P_SAMPLING] Using greedy selection")
            selected_token = sorted_indices[..., 0:1]
        else:
            print(f"[TOP_P_SAMPLING] Final probs sum: {sorted_probs_filtered.sum(dim=-1).item():.6f}")
            print(f"[TOP_P_SAMPLING] Non-zero elements: {(sorted_probs_filtered > 1e-10).sum().item()}")
            
            # Step 6: Sample with error handling
            try:
                # Double-check the tensor before sampling
                if sorted_probs_filtered.sum() == 0:
                    print(f"[TOP_P_SAMPLING] Zero sum probabilities, using greedy")
                    selected_token = sorted_indices[..., 0:1]
                else:
                    # Sample from the filtered distribution
                    sample_idx = torch.multinomial(sorted_probs_filtered, num_samples=1)
                    selected_token = sorted_indices.gather(-1, sample_idx)
                    
            except Exception as e:
                print(f"[TOP_P_SAMPLING] Sampling failed with error: {e}")
                print(f"[TOP_P_SAMPLING] Tensor stats - min: {sorted_probs_filtered.min()}, max: {sorted_probs_filtered.max()}, sum: {sorted_probs_filtered.sum()}")
                print(f"[TOP_P_SAMPLING] Falling back to greedy selection")
                selected_token = sorted_indices[..., 0:1]
    
    # Ensure consistent output shape [1, 1]
    if selected_token.dim() == 1:
        selected_token = selected_token.unsqueeze(0)
    elif selected_token.dim() == 0:
        selected_token = selected_token.unsqueeze(0).unsqueeze(0)
    
    print(f"[TOP_P_SAMPLING] Successfully sampled token: {selected_token.squeeze().item()}")
    print(f"[TOP_P_SAMPLING] Output token shape: {selected_token.shape}")
    return selected_token


def paligemma_forward_single_token(
    model: PaliGemmaForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kv_cache: Optional[KVCache] = None,
    pixel_values: Optional[torch.Tensor] = None
):
    """Forward pass for a single token - used in speculative decoding."""
    if input_ids.dim() == 1:  
        input_ids = input_ids.unsqueeze(0)   # Ensure shape [B, T]
    
    print(f"[FORWARD_SINGLE] Input IDs shape: {input_ids.shape}, attention mask shape: {attention_mask.shape}")
    print(f"[FORWARD_SINGLE] Has pixel values: {pixel_values is not None}")
    print(f"[FORWARD_SINGLE] KV cache items: {kv_cache.num_items() if kv_cache else 0}")
    
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
    )
    
    print(f"[FORWARD_SINGLE] Output logits shape: {outputs['logits'].shape}")
    return outputs


def decode_n_tokens_draft(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    cur_token: torch.Tensor,
    attention_mask: torch.Tensor,
    kv_cache: KVCache,
    num_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 0.9,
    do_sample: bool = True
):
    """Decode n tokens using draft model sequentially with fixed attention mask handling."""
    print(f"[DECODE_N_TOKENS_DRAFT] Generating {num_new_tokens} tokens with draft model")
    print(f"[DECODE_N_TOKENS_DRAFT] Current token: {cur_token}, shape: {cur_token.shape}")
    print(f"[DECODE_N_TOKENS_DRAFT] Initial attention mask shape: {attention_mask.shape}")
    print(f"[DECODE_N_TOKENS_DRAFT] Initial KV cache items: {kv_cache.num_items()}")
    
    new_tokens = []
    new_probs = []
    current_input_ids = cur_token.clone()
    current_attention_mask = attention_mask.clone()
    
    for i in range(num_new_tokens):
        print(f"[DECODE_N_TOKENS_DRAFT] Draft token {i+1}/{num_new_tokens}")
        print(f"[DECODE_N_TOKENS_DRAFT] Current attention mask shape: {current_attention_mask.shape}")
        print(f"[DECODE_N_TOKENS_DRAFT] Current KV cache items: {kv_cache.num_items()}")
        
        # CRITICAL: Ensure attention mask length matches KV cache + 1 (for current token)
        expected_mask_len = kv_cache.num_items() + 1
        if current_attention_mask.shape[1] != expected_mask_len:
            print(f"[DECODE_N_TOKENS_DRAFT] Attention mask size mismatch: {current_attention_mask.shape[1]} != {expected_mask_len}")
            if current_attention_mask.shape[1] < expected_mask_len:
                # Pad attention mask
                pad_len = expected_mask_len - current_attention_mask.shape[1]
                padding = torch.ones((1, pad_len), device=current_attention_mask.device)
                current_attention_mask = torch.cat([current_attention_mask, padding], dim=-1)
                print(f"[DECODE_N_TOKENS_DRAFT] Padded attention mask to {current_attention_mask.shape}")
            else:
                # Truncate attention mask
                current_attention_mask = current_attention_mask[:, :expected_mask_len]
                print(f"[DECODE_N_TOKENS_DRAFT] Truncated attention mask to {current_attention_mask.shape}")
        
        # Forward pass through draft model
        outputs = paligemma_forward_single_token(
            model, current_input_ids, current_attention_mask, kv_cache, pixel_values=None
        )
        
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        
        # Sample next token
        if do_sample:
            if top_p < 1.0:
                next_token = _sample_top_p(next_token_logits / temperature, top_p)
                # We need probabilities for speculative decoding - compute them from logits
                probs = logits_to_probs(outputs["logits"][:, -1:, :], temperature, top_k)
                new_probs.append(probs)
            else:
                next_token, next_prob = sample_token(outputs["logits"], temperature, top_k)
                new_probs.append(next_prob)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # For greedy decoding, we still need probabilities for speculative decoding
            probs = logits_to_probs(outputs["logits"][:, -1:, :], temperature, top_k)
            new_probs.append(probs)
        
        print(f"[DECODE_N_TOKENS_DRAFT] Generated token {i+1}: {next_token.item()}")
        new_tokens.append(next_token.clone())
        
        # Update for next iteration
        current_input_ids = next_token
        # Extend attention mask for the next token
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), device=current_attention_mask.device)], dim=-1
        )
    
    print(f"[DECODE_N_TOKENS_DRAFT] Generated {len(new_tokens)} draft tokens")
    print(f"[DECODE_N_TOKENS_DRAFT] Final KV cache items: {kv_cache.num_items()}")
    return new_tokens, new_probs, kv_cache


def speculative_decode_paligemma(
    target_model: PaliGemmaForConditionalGeneration,
    draft_model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    cur_token: torch.Tensor,
    attention_mask: torch.Tensor,
    target_kv_cache: KVCache,
    draft_kv_cache: KVCache,
    speculate_k: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 0.9,
    do_sample: bool = True
) -> Tuple[torch.Tensor, KVCache, KVCache, int]:
    """Fixed speculative decoding for PaliGemma models with proper attention mask handling."""
    print(f"[SPECULATIVE_DECODE] Starting speculative decode with k={speculate_k}")
    print(f"[SPECULATIVE_DECODE] Current token: {cur_token.item()}")
    print(f"[SPECULATIVE_DECODE] Target KV cache items: {target_kv_cache.num_items()}")
    print(f"[SPECULATIVE_DECODE] Draft KV cache items: {draft_kv_cache.num_items()}")
    print(f"[SPECULATIVE_DECODE] Input attention mask shape: {attention_mask.shape}")
    
    device = cur_token.device

    # --- FIX RE-ADDED HERE ---
    # CRITICAL FIX: Extend attention mask for the current token FIRST
    # The DRAFT model needs this extended mask to process cur_token and predict the *next* token.
    current_attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)
    print(f"[SPECULATIVE_DECODE] Extended attention mask shape: {current_attention_mask.shape}")
    print(f"[SPECULATIVE_DECODE] Expected mask length for current token: {target_kv_cache.num_items() + 1}")
    # --- END FIX ---
    
    # (The original buggy "Step 1: Add the current token to target KV cache" block remains correctly DELETED)

    # Step 2: Generate k draft tokens sequentially using draft model
    # Use the extended attention mask for draft model too
    draft_tokens, draft_probs, updated_draft_kv_cache = decode_n_tokens_draft(
        draft_model, processor, cur_token, current_attention_mask, draft_kv_cache, 
        speculate_k, temperature, top_k, top_p, do_sample
    )
    
    # Extract token values and create tensor
    draft_token_values = []
    for token in draft_tokens:
        if token.numel() == 1:
            draft_token_values.append(token.item())
        else:
            raise ValueError(f"Expected single-element tensor, got {token.shape}")
    
    draft_tokens_tensor = torch.tensor(draft_token_values, device=device, dtype=torch.long)
    print(f"[SPECULATIVE_DECODE] Draft tokens: {draft_tokens_tensor.tolist()}")
    
    # Step 3: Run target model on ONLY the draft tokens
    target_input_sequence = torch.tensor([draft_token_values], device=device, dtype=torch.long)
    print(f"[SPECULATIVE_DECODE] Target input sequence shape: {target_input_sequence.shape}")
    
    # Create attention mask for the draft tokens
    # Current state: target_kv_cache has processed [original_sequence + current_token]
    # We need mask for [original_sequence + current_token + draft_tokens]
    expected_target_mask_len = target_kv_cache.num_items() + target_input_sequence.shape[1]
    target_attention_mask = current_attention_mask.clone()
    
    # Extend attention mask for draft tokens
    if target_attention_mask.shape[1] < expected_target_mask_len:
        pad_len = expected_target_mask_len - target_attention_mask.shape[1]
        padding = torch.ones((1, pad_len), device=device)
        target_attention_mask = torch.cat([target_attention_mask, padding], dim=-1)
        print(f"[SPECULATIVE_DECODE] Extended target attention mask to {target_attention_mask.shape}")
    elif target_attention_mask.shape[1] > expected_target_mask_len:
        target_attention_mask = target_attention_mask[:, :expected_target_mask_len]
        print(f"[SPECULATIVE_DECODE] Truncated target attention mask to {target_attention_mask.shape}")
    
    print(f"[SPECULATIVE_DECODE] Final target attention mask shape: {target_attention_mask.shape}")
    print(f"[SPECULATIVE_DECODE] Expected target mask length: {expected_target_mask_len}")
    
    # Run target model on draft tokens
    target_outputs = paligemma_forward_single_token(
        target_model, target_input_sequence, target_attention_mask, target_kv_cache, pixel_values=None
    )
    
    target_logits = target_outputs["logits"]  # Shape: [1, seq_len, vocab_size]
    updated_target_kv_cache = target_outputs["kv_cache"]
    
    # Convert to probabilities
    target_probs = logits_to_probs(target_logits[0], temperature, top_k)  # Shape: [seq_len, vocab_size]
    print(f"[SPECULATIVE_DECODE] Target probs shape: {target_probs.shape}")
    
    # Stack draft probabilities for comparison
    if len(draft_probs) > 0:
        # Handle different probability tensor shapes
        draft_probs_list = []
        for p in draft_probs:
            if p.dim() == 3:  # Shape: [1, 1, vocab_size]
                draft_probs_list.append(p.squeeze(0).squeeze(0))  # -> [vocab_size]
            elif p.dim() == 2:  # Shape: [1, vocab_size]
                draft_probs_list.append(p.squeeze(0))  # -> [vocab_size]
            else:  # Shape: [vocab_size]
                draft_probs_list.append(p)
        draft_probs_tensor = torch.stack(draft_probs_list)  # Shape: [speculate_k, vocab_size]
        print(f"[SPECULATIVE_DECODE] Draft probs tensor shape: {draft_probs_tensor.shape}")
    else:
        # Fallback if no draft probs (shouldn't happen in normal execution)
        draft_probs_tensor = torch.ones((speculate_k, target_probs.shape[-1]), device=device) / target_probs.shape[-1]
    
    # Step 4: Acceptance/rejection phase
    accept_length = 0
    accepted_tokens = []
    
    for i in range(speculate_k):
        draft_token = draft_tokens_tensor[i].item()
        
        # target_probs[i] corresponds to draft_tokens[i]
        target_prob_at_pos = target_probs[i]  # Target probabilities at position i
        draft_prob_at_pos = draft_probs_tensor[i]  # Draft probabilities at position i
        
        # Get probabilities for the draft token
        p_draft = draft_prob_at_pos[draft_token].item()
        q_target = target_prob_at_pos[draft_token].item()
        
        print(f"[SPECULATIVE_DECODE] Token {i}: draft_token={draft_token}, p_draft={p_draft:.6f}, q_target={q_target:.6f}")
        
        # Acceptance condition: min(1, q/p)
        if p_draft > 0:
            accept_prob = min(1.0, q_target / p_draft)
        else:
            accept_prob = 0.0
        
        print(f"[SPECULATIVE_DECODE] Acceptance probability: {accept_prob:.6f}")
        
        # Sample to decide acceptance
        if torch.rand(1).item() < accept_prob:
            print(f"[SPECULATIVE_DECODE] ✓ Accepted token {i}: {draft_token}")
            accepted_tokens.append(torch.tensor([draft_token], device=device))
            accept_length += 1
        else:
            print(f"[SPECULATIVE_DECODE] ✗ Rejected token {i}: {draft_token}")
            # Sample from corrected distribution: max(0, q - p) / sum(max(0, q - p))
            corrected_probs = torch.clamp(target_prob_at_pos - draft_prob_at_pos, min=0.0)
            corrected_probs_sum = corrected_probs.sum()
            
            if corrected_probs_sum > 0:
                corrected_probs = corrected_probs / corrected_probs_sum
                corrected_token = torch.multinomial(corrected_probs, num_samples=1)
                print(f"[SPECULATIVE_DECODE] Sampled corrected token: {corrected_token.item()}")
            else:
                # Fallback: sample from target distribution
                corrected_token = torch.multinomial(target_prob_at_pos, num_samples=1)
                print(f"[SPECULATIVE_DECODE] Fallback: sampled from target distribution: {corrected_token.item()}")
            
            accepted_tokens.append(corrected_token)
            break
            
##uncomment above for corrrect fix            
            
            
  
   
    
    # If all tokens were accepted, sample one more from the target model
    if accept_length == speculate_k:
        print(f"[SPECULATIVE_DECODE] All {speculate_k} tokens accepted! Sampling bonus token.")
        # Bug fix: Should sample from the *last* set of target logits
        # target_probs shape is [speculate_k, vocab_size]. We need the probs calculated for the *last* draft token.
        # target_logits[0] was shape [speculate_k, vocab_size], so target_probs is the same.
        # target_probs[speculate_k-1] or target_probs[-1] is correct.
        bonus_probs = target_probs[-1]  # Use last position's probabilities
        bonus_token = torch.multinomial(bonus_probs, num_samples=1)
        accepted_tokens.append(bonus_token)
        accept_length += 1
        print(f"[SPECULATIVE_DECODE] Bonus token: {bonus_token.item()}")

    # Concatenate accepted tokens - ensure consistent shape
    if accepted_tokens:
        # Make sure all tokens have consistent shape [1]
        normalized_tokens = []
        for token in accepted_tokens:
            if token.dim() == 0:
                normalized_tokens.append(token.unsqueeze(0))
            elif token.dim() == 1:
                normalized_tokens.append(token)
            else:
                normalized_tokens.append(token.squeeze())
        final_tokens = torch.cat(normalized_tokens, dim=0)
    else:
        # This shouldn't happen, but as a safeguard (e.g., if draft model fails to produce tokens)
        # We must return *something* to avoid a crash. Let's sample from the target model's *first* prediction (based on cur_token)
        # NOTE: This logic path is now broken because we removed the initial target pass.
        # If accepted_tokens is empty, it means the loop broke on the first rejection (i=0)
        # and it *should* contain the single 'corrected_token'. This 'else' path is likely unreachable.
        # But just in case, let's create an empty tensor. The main loop logic will handle it.
        # Actually, the rejection logic *always* appends a token, so 'accepted_tokens' will never be empty if speculate_k > 0.
        # This path is safe.
        final_tokens = torch.tensor([], device=device, dtype=torch.long)  # Should be handled by rejection logic
        print("[SPECULATIVE_DECODE] WARNING: No tokens were accepted or corrected.")

    print(f"[SPECULATIVE_DECODE] Final accepted tokens: {final_tokens.tolist()}")
    print(f"[SPECULATIVE_DECODE] Acceptance rate: {len(final_tokens)} tokens generated ({accept_length-1} accepted / {speculate_k} speculated)")  # A bit confusing, let's stick to your old log
    print(f"[SPECULATIVE_DECODE] Acceptance count: {accept_length} (includes bonus/corrected)")
    print(f"[SPECULATIVE_DECODE] Final target KV cache items: {updated_target_kv_cache.num_items()}")
    print(f"[SPECULATIVE_DECODE] Final draft KV cache items: {updated_draft_kv_cache.num_items()}")
    
    # The number accepted for stats should be the number *actually accepted* (accept_length)
    # or if a rejection happened, it's the index 'i' where it broke + 1 (for the corrected token).
    # The 'num_accepted' should just be the count of tokens we are returning.
    num_accepted_for_stats = len(final_tokens)
    
    return final_tokens, updated_target_kv_cache, updated_draft_kv_cache, num_accepted_for_stats


def test_speculative_inference(
    target_model: PaliGemmaForConditionalGeneration,
    draft_model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompts: List[str],
    image_file_paths: List[str],
    max_tokens_to_generate: int,
    speculate_k: int = 4,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """Main function for speculative inference with PaliGemma."""
    print(f"[SPEC_INFERENCE] Starting speculative inference")
    print(f"[SPEC_INFERENCE] Max tokens: {max_tokens_to_generate}, speculate_k: {speculate_k}")
    print(f"[SPEC_INFERENCE] Temperature: {temperature}, top_p: {top_p}, do_sample: {do_sample}")
    
    # Get initial inputs (only for the first batch item for simplicity)
    model_inputs = get_model_inputs(processor, [prompts[0]], [image_file_paths[0]], device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    
    print(f"[SPEC_INFERENCE] Initial input_ids shape: {input_ids.shape}")
    print(f"[SPEC_INFERENCE] Initial attention_mask shape: {attention_mask.shape}")
    print(f"[SPEC_INFERENCE] Initial pixel_values shape: {pixel_values.shape}")
    
    # Initialize KV caches
    target_kv_cache = KVCache()
    draft_kv_cache = KVCache()
    
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    acceptance_counts = [0] * (speculate_k + 2)  # Track acceptance statistics
    
    # FIXED: Single prefill phase for both models
    print(f"[SPEC_INFERENCE] === PREFILL PHASE ===")
    
    # Target model prefill
    print(f"[SPEC_INFERENCE] Target model prefill...")
    target_outputs = target_model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=target_kv_cache,
    )
    target_kv_cache = target_outputs["kv_cache"]
    
    # Draft model prefill
    print(f"[SPEC_INFERENCE] Draft model prefill...")
    draft_outputs = draft_model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=draft_kv_cache,
    )
    draft_kv_cache = draft_outputs["kv_cache"]
    
    # Sample first token from target model only
    next_token_logits = target_outputs["logits"][:, -1, :]
    if do_sample:
        next_token = _sample_top_p(next_token_logits / temperature, top_p)
    else:
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    print(f"[SPEC_INFERENCE] First generated token: {next_token.item()}")
    
    # Ensure consistent shape for generated_tokens
    if next_token.dim() == 2:
        generated_tokens.append(next_token.squeeze(0))  # Convert [1, 1] -> [1]
    else:
        generated_tokens.append(next_token)
    
    # Update attention mask for generated token (this will be the base for speculative decoding)
    current_token = next_token.squeeze() if next_token.dim() > 1 else next_token
    if current_token.dim() == 0:
        current_token = current_token.unsqueeze(0)
    
    print(f"[SPEC_INFERENCE] === SPECULATIVE DECODING PHASE ===")
    
    # Speculative decoding loop
    for step in range(1, max_tokens_to_generate):
        print(f"\n[SPEC_INFERENCE] === Step {step}/{max_tokens_to_generate-1} ===")
        
        if current_token.item() == stop_token:
            print(f"[SPEC_INFERENCE] Hit stop token, ending generation")
            break
        
        # Perform speculative decoding
        start_time = time.time()
        accepted_tokens, target_kv_cache, draft_kv_cache, num_accepted = speculative_decode_paligemma(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            cur_token=current_token,
            attention_mask=attention_mask,
            target_kv_cache=target_kv_cache,
            draft_kv_cache=draft_kv_cache,
            speculate_k=speculate_k,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )
        decode_time = time.time() - start_time
        
        # Update statistics
        acceptance_counts[num_accepted] += 1
        
        print(f"[SPEC_INFERENCE] Step {step} completed in {decode_time:.3f}s")
        print(f"[SPEC_INFERENCE] Accepted {num_accepted} tokens: {accepted_tokens.tolist()}")
        
        # Add accepted tokens to generated sequence - ensure consistent shapes
        for i, token_val in enumerate(accepted_tokens.tolist()):
            token_tensor = torch.tensor([token_val], device=device)
            generated_tokens.append(token_tensor)
            # Extend attention mask for each accepted token
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)
        
        # Update current token for next iteration
        if accepted_tokens.numel() > 0:
            current_token = torch.tensor([accepted_tokens[-1].item()], device=device)
        else:
            # This shouldn't happen, but as a safeguard
            current_token = torch.tensor([stop_token], device=device)
        
        # Check if we've generated enough tokens
        if len(generated_tokens) >= max_tokens_to_generate:
            print(f"[SPEC_INFERENCE] Generated {len(generated_tokens)} tokens, stopping")
            break
        
        # Check for stop token in accepted tokens
        if any(token_val == stop_token for token_val in accepted_tokens.tolist()):
            print(f"[SPEC_INFERENCE] Stop token found in accepted tokens, ending generation")
            break
    
    # Decode final output - ensure all tokens have consistent shape
    if generated_tokens:
        # Normalize all tokens to have shape [1]
        normalized_tokens = []
        for token in generated_tokens:
            if token.dim() == 0:
                normalized_tokens.append(token.unsqueeze(0))
            elif token.dim() == 1:
                normalized_tokens.append(token)
            else:
                # Handle 2D case by squeezing
                normalized_tokens.append(token.squeeze())
        
        final_tokens = torch.cat(normalized_tokens, dim=0)
        decoded_output = processor.tokenizer.decode(final_tokens, skip_special_tokens=True)
    else:
        decoded_output = "(No tokens generated)"
    
    # Print statistics
    total_steps = sum(acceptance_counts)
    print(f"\n[SPEC_INFERENCE] === FINAL STATISTICS ===")
    print(f"[SPEC_INFERENCE] Total generation steps: {total_steps}")
    print(f"[SPEC_INFERENCE] Acceptance distribution:")
    for i, count in enumerate(acceptance_counts):
        if count > 0:
            percentage = (count / total_steps) * 100 if total_steps > 0 else 0
            print(f"[SPEC_INFERENCE]   {i} tokens accepted: {count} times ({percentage:.1f}%)")
    
    # Calculate average acceptance rate
    if total_steps > 0:
        avg_acceptance = sum(i * count for i, count in enumerate(acceptance_counts)) / total_steps
        print(f"[SPEC_INFERENCE] Average tokens accepted per step: {avg_acceptance:.2f}")
        efficiency_gain = avg_acceptance / 1.0  # Compared to standard decoding
        print(f"[SPEC_INFERENCE] Theoretical speedup: {efficiency_gain:.2f}x")
    
    print(f"\n[SPEC_INFERENCE] === FINAL OUTPUT ===")
    print(f"Prompt: {prompts[0]}")
    print(f"Generated: {decoded_output}")
    print("=" * 80)


def main(
    target_model_path: str = None,
    draft_model_path: str = None,
    prompt: str = "What is shown in this image?",
    image_file_path: str = None,
    max_tokens_to_generate: int = 50,
    speculate_k: int = 4,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: float = 0.9,
    do_sample: bool = True,
    only_cpu: bool = False,
):
    """Main function to run speculative decoding."""
    print("Set manual seed")
    torch.manual_seed(42) 
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)
    print(f"Target model: {target_model_path}")
    print(f"Draft model: {draft_model_path}")

    # Load models
    print("Loading target model...")
    target_model, tokenizer = load_hf_model(target_model_path, device)
    target_model = target_model.to(device).eval()
    
    print("Loading draft model from local...")
    draft_model, _ = load_hf_model(draft_model_path, device)
    draft_model = draft_model.to(device).eval()

    if device == "cuda":
        target_model = target_model.to(torch.bfloat16)
        draft_model = draft_model.to(torch.bfloat16)
        print("Models converted to bfloat16")

    # Create processor
    num_image_tokens = target_model.config.vision_config.num_image_tokens
    image_size = target_model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print(f"Running speculative inference...")
    print(f"Prompt: '{prompt}'")
    print(f"Image: {image_file_path}")
    print(f"Max tokens: {max_tokens_to_generate}")
    print(f"Speculate k: {speculate_k}")
    
    with torch.no_grad():
        test_speculative_inference(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            device=device,
            prompts=[prompt],
            image_file_paths=[image_file_path],
            max_tokens_to_generate=max_tokens_to_generate,
            speculate_k=speculate_k,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)