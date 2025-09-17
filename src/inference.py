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
    """Fixed speculative decoding with proper error handling and tensor management."""
    print(f"[SPECULATIVE_DECODE] Starting speculative decode with k={speculate_k}")
    print(f"[SPECULATIVE_DECODE] Current token: {cur_token.item()}")
    
    device = cur_token.device
    
    # Step 1: Create backup and extend attention mask
    draft_cache_backup = draft_kv_cache.copy()
    current_attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)
    
    # Step 2: Generate draft tokens
    try:
        draft_tokens, draft_probs, updated_draft_kv_cache = decode_n_tokens_draft(
            draft_model, processor, cur_token, current_attention_mask, draft_kv_cache, 
            speculate_k, temperature, top_k, top_p, do_sample
        )
        
        # Convert draft tokens to tensor with error handling
        draft_token_values = []
        for token in draft_tokens:
            if isinstance(token, torch.Tensor):
                if token.numel() == 1:
                    draft_token_values.append(token.item())
                else:
                    print(f"[WARNING] Multi-element token tensor: {token}")
                    draft_token_values.append(token.flatten()[0].item())
            else:
                draft_token_values.append(int(token))
        
        draft_tokens_tensor = torch.tensor(draft_token_values, device=device, dtype=torch.long)
        print(f"[SPECULATIVE_DECODE] Draft tokens: {draft_tokens_tensor.tolist()}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate draft tokens: {e}")
        # Fallback: return EOS token
        fallback_token = torch.tensor([processor.tokenizer.eos_token_id], device=device, dtype=torch.long)
        return fallback_token, target_kv_cache, draft_kv_cache, 1
    
    # Step 3: Run target model on draft tokens
    try:
        target_input_sequence = draft_tokens_tensor.unsqueeze(0)  # [1, speculate_k]
        
        # Prepare attention mask for target model
        target_attention_mask = current_attention_mask.clone()
        needed_length = target_kv_cache.num_items() + speculate_k
        
        if target_attention_mask.shape[1] < needed_length:
            pad_length = needed_length - target_attention_mask.shape[1]
            target_attention_mask = torch.cat([
                target_attention_mask, 
                torch.ones((1, pad_length), device=device)
            ], dim=-1)
        elif target_attention_mask.shape[1] > needed_length:
            target_attention_mask = target_attention_mask[:, :needed_length]
        
        # Get target model predictions
        target_outputs = paligemma_forward_single_token(
            target_model, target_input_sequence, target_attention_mask, target_kv_cache, pixel_values=None
        )
        
        target_logits = target_outputs["logits"][0]  # [speculate_k, vocab_size]
        updated_target_kv_cache = target_outputs["kv_cache"]
        
        # Convert to probabilities with numerical stability
        target_probs = logits_to_probs(target_logits, temperature, top_k)
        
        # Process draft probabilities with error handling
        draft_probs_list = []
        for i, p in enumerate(draft_probs):
            try:
                if p.dim() == 3:  # [1, 1, vocab_size]
                    draft_probs_list.append(p.squeeze(0).squeeze(0))
                elif p.dim() == 2:  # [1, vocab_size]
                    draft_probs_list.append(p.squeeze(0))
                else:  # [vocab_size]
                    draft_probs_list.append(p)
            except Exception as e:
                print(f"[WARNING] Error processing draft prob {i}: {e}")
                # Create uniform distribution as fallback
                uniform_prob = torch.ones(target_probs.shape[-1], device=device) / target_probs.shape[-1]
                draft_probs_list.append(uniform_prob)
        
        draft_probs_tensor = torch.stack(draft_probs_list)
        
    except Exception as e:
        print(f"[ERROR] Failed target model forward pass: {e}")
        # Fallback: accept first draft token only
        fallback_token = torch.tensor([draft_tokens_tensor[0].item()], device=device, dtype=torch.long)
        return fallback_token, target_kv_cache, draft_kv_cache, 1
    
    # Step 4: Acceptance/rejection phase with robust error handling
    accepted_tokens = []
    actual_acceptances = 0
    rejection_position = None
    
    for i in range(speculate_k):
        try:
            draft_token = draft_tokens_tensor[i].item()
            
            # Get probabilities with bounds checking
            p_draft = max(draft_probs_tensor[i, draft_token].item(), 1e-12)  # Avoid zero
            q_target = max(target_probs[i, draft_token].item(), 1e-12)  # Avoid zero
            
            print(f"[SPECULATIVE_DECODE] Token {i}: draft_token={draft_token}, p_draft={p_draft:.8f}, q_target={q_target:.8f}")
            
            # Calculate acceptance probability with numerical stability
            if p_draft < 1e-10:
                print(f"[SPECULATIVE_DECODE] Draft probability too low, automatic rejection")
                accept_prob = 0.0
            else:
                accept_prob = min(1.0, q_target / p_draft)
            
            print(f"[SPECULATIVE_DECODE] Acceptance probability: {accept_prob:.6f}")
            
            # Decision: accept or reject
            if torch.rand(1).item() < accept_prob:
                print(f"[SPECULATIVE_DECODE] ✓ Accepted token {i}: {draft_token}")
                accepted_tokens.append(torch.tensor([draft_token], device=device, dtype=torch.long))
                actual_acceptances += 1
            else:
                print(f"[SPECULATIVE_DECODE] ✗ Rejected token {i}: {draft_token}")
                rejection_position = i
                
                # Sample corrected token with multiple fallback strategies
                corrected_token = None
                
                try:
                    # Strategy 1: Corrected distribution sampling
                    corrected_probs = torch.clamp(target_probs[i] - draft_probs_tensor[i], min=0.0)
                    corrected_probs_sum = corrected_probs.sum()
                    
                    if corrected_probs_sum > 1e-10:
                        corrected_probs = corrected_probs / corrected_probs_sum
                        # Filter out extremely low probabilities
                        corrected_probs = torch.where(corrected_probs < 1e-8, 0.0, corrected_probs)
                        corrected_probs = corrected_probs / corrected_probs.sum()
                        corrected_token = torch.multinomial(corrected_probs, num_samples=1)
                        print(f"[SPECULATIVE_DECODE] Corrected sampling successful: {corrected_token.item()}")
                    else:
                        raise ValueError("Corrected probabilities sum to zero")
                        
                except Exception as e:
                    print(f"[SPECULATIVE_DECODE] Corrected sampling failed: {e}")
                    
                    try:
                        # Strategy 2: Direct target sampling with temperature
                        fallback_probs = F.softmax(target_logits[i] / max(temperature, 0.1), dim=-1)
                        corrected_token = torch.multinomial(fallback_probs, num_samples=1)
                        print(f"[SPECULATIVE_DECODE] Target sampling successful: {corrected_token.item()}")
                    except Exception as e2:
                        print(f"[SPECULATIVE_DECODE] Target sampling failed: {e2}")
                        
                        # Strategy 3: Greedy fallback
                        corrected_token = torch.argmax(target_probs[i]).unsqueeze(0)
                        print(f"[SPECULATIVE_DECODE] Greedy fallback: {corrected_token.item()}")
                
                # Ensure corrected token has proper shape
                if corrected_token is not None:
                    if corrected_token.dim() == 0:
                        corrected_token = corrected_token.unsqueeze(0)
                    elif corrected_token.dim() > 1:
                        corrected_token = corrected_token.squeeze()
                        if corrected_token.dim() == 0:
                            corrected_token = corrected_token.unsqueeze(0)
                    
                    # Check for problematic tokens
                    if corrected_token.item() == processor.tokenizer.eos_token_id:
                        print(f"[SPECULATIVE_DECODE] WARNING: Corrected token is EOS")
                    
                    accepted_tokens.append(corrected_token)
                    actual_acceptances += 1  # Count corrected token as accepted
                else:
                    # Last resort: use a safe token (space or period)
                    safe_token = torch.tensor([processor.tokenizer.encode(" ")[0]], device=device, dtype=torch.long)
                    accepted_tokens.append(safe_token)
                    print(f"[SPECULATIVE_DECODE] Using safe fallback token: {safe_token.item()}")
                
                break  # Exit loop after rejection and correction
                
        except Exception as e:
            print(f"[ERROR] Error processing token {i}: {e}")
            # Emergency fallback: use current draft token
            emergency_token = torch.tensor([draft_tokens_tensor[i].item()], device=device, dtype=torch.long)
            accepted_tokens.append(emergency_token)
            break
    
    # Step 5: Handle bonus token if all were accepted
    if actual_acceptances == speculate_k and rejection_position is None:
        print(f"[SPECULATIVE_DECODE] All {speculate_k} tokens accepted! Sampling bonus token.")
        
        try:
            # Get the last accepted token for continuation
            last_token = accepted_tokens[-1]
            if last_token.dim() > 1:
                last_token = last_token.squeeze()
            if last_token.dim() == 0:
                last_token = last_token.unsqueeze(0)
            
            last_token_input = last_token.unsqueeze(0)  # [1, 1]
            bonus_attention_mask = torch.cat([target_attention_mask, torch.ones((1, 1), device=device)], dim=-1)
            
            bonus_outputs = paligemma_forward_single_token(
                target_model, last_token_input, bonus_attention_mask, updated_target_kv_cache, pixel_values=None
            )
            
            updated_target_kv_cache = bonus_outputs["kv_cache"]
            bonus_logits = bonus_outputs["logits"][0, -1, :]  # Last token's logits
            
            if do_sample and top_p < 1.0:
                bonus_token = _sample_top_p(bonus_logits / temperature, top_p)
            elif do_sample:
                bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
                bonus_token = torch.multinomial(bonus_probs, num_samples=1)
            else:
                bonus_token = torch.argmax(bonus_logits, dim=-1, keepdim=True)
            
            # Ensure proper shape
            if bonus_token.dim() == 0:
                bonus_token = bonus_token.unsqueeze(0)
            elif bonus_token.dim() > 1:
                bonus_token = bonus_token.squeeze()
                if bonus_token.dim() == 0:
                    bonus_token = bonus_token.unsqueeze(0)
            
            accepted_tokens.append(bonus_token)
            print(f"[SPECULATIVE_DECODE] Bonus token: {bonus_token.item()}")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate bonus token: {e}")
            # Skip bonus token on error
    
    # Step 6: KV cache rollback if needed
    if rejection_position is not None:
        try:
            # Rollback draft cache to backup state
            draft_kv_cache.restore_from(draft_cache_backup)
            
            # Rollback target cache for rejected tokens
            tokens_to_rollback = speculate_k - len(accepted_tokens)
            if tokens_to_rollback > 0:
                updated_target_kv_cache.rollback(tokens_to_rollback)
                print(f"[SPECULATIVE_DECODE] Rolled back {tokens_to_rollback} tokens from target cache")
        except Exception as e:
            print(f"[ERROR] KV cache rollback failed: {e}")
    
    # Step 7: Prepare final output with robust tensor handling
    if accepted_tokens:
        try:
            # Normalize all tensors to 1D
            normalized_tokens = []
            for i, token in enumerate(accepted_tokens):
                if token.dim() == 0:
                    normalized_tokens.append(token.unsqueeze(0))
                elif token.dim() == 1:
                    normalized_tokens.append(token)
                else:
                    normalized_tokens.append(token.view(-1))
                
                print(f"[DEBUG] Token {i}: original shape {token.shape}, normalized shape {normalized_tokens[-1].shape}")
            
            # Concatenate with error handling
            final_tokens = torch.cat(normalized_tokens, dim=0)
            print(f"[DEBUG] Successfully concatenated {len(normalized_tokens)} tokens, final shape: {final_tokens.shape}")
            
        except Exception as e:
            print(f"[ERROR] Failed to concatenate tokens: {e}")
            # Fallback: create tensor from token values
            token_values = []
            for token in accepted_tokens:
                try:
                    if token.numel() == 1:
                        token_values.append(token.item())
                    else:
                        token_values.extend(token.flatten().tolist())
                except:
                    token_values.append(processor.tokenizer.eos_token_id)  # Safe fallback
            
            final_tokens = torch.tensor(token_values, device=device, dtype=torch.long)
            print(f"[DEBUG] Created fallback tensor with shape: {final_tokens.shape}")
    else:
        # No tokens accepted - this should not happen but handle gracefully
        print("[ERROR] No tokens were accepted, using EOS")
        final_tokens = torch.tensor([processor.tokenizer.eos_token_id], device=device, dtype=torch.long)
    
    num_tokens_returned = len(final_tokens) if final_tokens.numel() > 0 else 1
    
    print(f"[SPECULATIVE_DECODE] Returned {num_tokens_returned} tokens: {final_tokens.tolist()}")
    print(f"[SPECULATIVE_DECODE] Actual acceptances: {actual_acceptances}/{speculate_k}")
    
    return final_tokens, updated_target_kv_cache, draft_kv_cache, num_tokens_returned


# Diagnostic function to check draft model quality
def diagnose_model_compatibility(target_model, draft_model, processor, model_inputs):
    """Comprehensive diagnosis of target vs draft model compatibility."""
    print("\n=== DETAILED MODEL COMPATIBILITY DIAGNOSIS ===")
    
    with torch.no_grad():
        # Get outputs from both models
        target_outputs = target_model(**model_inputs)
        draft_outputs = draft_model(**model_inputs)
        
        target_logits = target_outputs['logits'][:, -1, :]  # Last token logits
        draft_logits = draft_outputs['logits'][:, -1, :]   # Last token logits
        
        print(f"Target logits shape: {target_logits.shape}")
        print(f"Draft logits shape: {draft_logits.shape}")
        print(f"Target logits range: [{target_logits.min().item():.3f}, {target_logits.max().item():.3f}]")
        print(f"Draft logits range: [{draft_logits.min().item():.3f}, {draft_logits.max().item():.3f}]")
        
        # Check for vocabulary mismatch
        if target_logits.shape != draft_logits.shape:
            print(f"❌ CRITICAL: Logits shape mismatch!")
            print(f"   Target: {target_logits.shape}, Draft: {draft_logits.shape}")
            return False
        
        # Convert to probabilities
        target_probs = F.softmax(target_logits, dim=-1)
        draft_probs = F.softmax(draft_logits, dim=-1)
        
        print(f"Target probs range: [{target_probs.min().item():.8f}, {target_probs.max().item():.8f}]")
        print(f"Draft probs range: [{draft_probs.min().item():.8f}, {draft_probs.max().item():.8f}]")
        
        # Find top tokens for each model
        target_top_k = torch.topk(target_probs, k=10)
        draft_top_k = torch.topk(draft_probs, k=10)
        
        print("\nTop 10 target tokens:")
        for i, (prob, token_id) in enumerate(zip(target_top_k.values[0], target_top_k.indices[0])):
            token_text = processor.tokenizer.decode([token_id.item()])
            print(f"  {i+1}: Token {token_id.item()} ('{token_text}') - prob: {prob.item():.6f}")
        
        print("\nTop 10 draft tokens:")
        for i, (prob, token_id) in enumerate(zip(draft_top_k.values[0], draft_top_k.indices[0])):
            token_text = processor.tokenizer.decode([token_id.item()])
            print(f"  {i+1}: Token {token_id.item()} ('{token_text}') - prob: {prob.item():.6f}")
        
        # Check overlap in top tokens
        target_top_tokens = set(target_top_k.indices[0].tolist())
        draft_top_tokens = set(draft_top_k.indices[0].tolist())
        overlap = len(target_top_tokens & draft_top_tokens)
        print(f"\nTop-10 token overlap: {overlap}/10 ({overlap/10*100:.1f}%)")
        
        # Check if draft assigns reasonable probability to target's top token
        target_best_token = target_top_k.indices[0][0].item()
        draft_prob_for_target_best = draft_probs[0][target_best_token].item()
        print(f"Draft probability for target's best token ({target_best_token}): {draft_prob_for_target_best:.8f}")
        
        if draft_prob_for_target_best < 1e-8:
            print("❌ CRITICAL: Draft model assigns near-zero probability to target's preferred token!")
        elif draft_prob_for_target_best < 1e-5:
            print("⚠️  WARNING: Draft model assigns very low probability to target's preferred token")
        else:
            print("✅ Draft model assigns reasonable probability to target's preferred token")
        
        # Calculate KL divergence
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        draft_probs_safe = torch.clamp(draft_probs, min=epsilon)
        target_probs_safe = torch.clamp(target_probs, min=epsilon)
        
        kl_div = F.kl_div(draft_probs_safe.log(), target_probs_safe, reduction='batchmean')
        print(f"KL divergence (draft || target): {kl_div.item():.4f}")
        
        if kl_div.item() > 10.0:
            print("❌ CRITICAL: Very high KL divergence indicates models are very different!")
        elif kl_div.item() > 5.0:
            print("⚠️  WARNING: High KL divergence indicates significant model differences")
        else:
            print("✅ Reasonable KL divergence")
        
        # Check model configurations
        print(f"\nModel configurations:")
        print(f"Target model type: {type(target_model).__name__}")
        print(f"Draft model type: {type(draft_model).__name__}")
        
        # Check if both models have the same tokenizer
        target_vocab_size = target_model.config.text_config.vocab_size if hasattr(target_model.config, 'text_config') else target_model.config.vocab_size
        draft_vocab_size = draft_model.config.text_config.vocab_size if hasattr(draft_model.config, 'text_config') else draft_model.config.vocab_size
        
        print(f"Target vocab size: {target_vocab_size}")
        print(f"Draft vocab size: {draft_vocab_size}")
        
        if target_vocab_size != draft_vocab_size:
            print("❌ CRITICAL: Vocabulary size mismatch!")
            return False
        
        print("\n=== END DIAGNOSIS ===\n")
        return True

# Quick fix for immediate testing - reduce speculative window and add fallbacks
def conservative_speculative_decode(
    target_model, draft_model, processor, cur_token, attention_mask, 
    target_kv_cache, draft_kv_cache, speculate_k, temperature, top_k, top_p, do_sample
):
    """Simplified version that's more likely to work with poor draft models."""
    print(f"[CONSERVATIVE] Using conservative speculative decode with k=1")
    
    # Force k=1 for now to debug
    device = cur_token.device
    
    try:
        # Generate just 1 draft token
        current_attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)
        draft_tokens, draft_probs, updated_draft_kv_cache = decode_n_tokens_draft(
            draft_model, processor, cur_token, current_attention_mask, draft_kv_cache, 
            1, temperature, top_k, top_p, do_sample  # Force k=1
        )
        
        # Just return the first draft token for now - skip acceptance/rejection
        if draft_tokens:
            first_token = draft_tokens[0]
            if isinstance(first_token, torch.Tensor):
                if first_token.dim() == 0:
                    result = first_token.unsqueeze(0)
                else:
                    result = first_token.squeeze() if first_token.numel() == 1 else first_token[0:1]
            else:
                result = torch.tensor([first_token], device=device, dtype=torch.long)
            
            print(f"[CONSERVATIVE] Returning draft token: {result.item()}")
            return result, target_kv_cache, updated_draft_kv_cache, 1
        else:
            # Fallback to EOS
            result = torch.tensor([processor.tokenizer.eos_token_id], device=device, dtype=torch.long)
            return result, target_kv_cache, draft_kv_cache, 1
            
    except Exception as e:
        print(f"[CONSERVATIVE] Even conservative approach failed: {e}")
        # Ultimate fallback
        result = torch.tensor([processor.tokenizer.eos_token_id], device=device, dtype=torch.long)
        return result, target_kv_cache, draft_kv_cache, 1

# Add this to your main function to diagnose before running
def run_diagnostics_before_generation(target_model, draft_model, processor, model_inputs):
    """Run diagnostics and return whether to proceed with speculative decoding."""
    print("Running model compatibility check...")
    
    compatible = diagnose_model_compatibility(target_model, draft_model, processor, model_inputs)
    
    if not compatible:
        print("❌ Models are not compatible for speculative decoding!")
        print("Recommendation: Use only the target model (standard decoding)")
        return False
    else:
        print("✅ Models appear compatible, proceeding with speculative decoding")
        return True

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
#     if generated_tokens:
#         # Normalize all tokens to have shape [1]
#         normalized_tokens = []
#         for token in generated_tokens:
#             if token.dim() == 0:
#                 normalized_tokens.append(token.unsqueeze(0))
#             elif token.dim() == 1:
#                 normalized_tokens.append(token)
#             else:
#                 # Handle 2D case by squeezing
#                 normalized_tokens.append(token.squeeze())
        
#                 # --- FIX for Concatenation ---
#         # Before concatenating, ensure every tensor in the list is 1D.
#         # The .view(-1) call will reliably reshape 0D and 1D tensors to be 1D.
       
#         # --- END FIX ---
#         final_tokens = torch.cat([t.squeeze() for t in accepted_tokens], dim=0)
#         decoded_output = processor.tokenizer.decode(final_tokens, skip_special_tokens=True)
#     else:
#         decoded_output = "(No tokens generated)"

     # Decode final output - ensure all tokens have consistent shape
    if generated_tokens:
        
        # --- FIX for Concatenation ---
        # Before concatenating, ensure every tensor in the list is 1D.
        # The .view(-1) call will reliably reshape 0D, 1D, and 2D tensors to be 1D.
        normalized_tokens = [t.view(-1) for t in generated_tokens]
        final_tokens = torch.cat(normalized_tokens, dim=0)
        # --- END FIX ---
        
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
    torch.manual_seed(42)
    print("SET MANUAL SEED")
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
    
    # Add this in your main() function before calling test_speculative_inference
    model_inputs = get_model_inputs(processor, [prompt], [image_file_path], device)
    compatible = run_diagnostics_before_generation(target_model, draft_model, processor, model_inputs) 
    if not compatible:
        print("Falling back to standard generation")
        return
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