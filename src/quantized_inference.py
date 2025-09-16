"""
Quantized inference script with robust Flash Attention handling and error recovery.
"""

from PIL import Image
import torch
import torch.nn.functional as F
import fire
from typing import List, Optional, Tuple
import time
import os
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing_paligemma import PaliGemmaProcessor
from gemma_flash import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

# Import the quantization functions
try:
    from quantization_utils import (
        QuantizationConfig,
        quantize_paligemma_with_monitoring,
        quantize_int8_with_monitoring,
        quantize_int4_with_monitoring, 
        quantize_nf4_with_monitoring,
        PerformanceMetrics,
        QuantizationPerformanceMonitor
    )
    USE_ENHANCED = True
    print("Using enhanced quantization with comprehensive performance monitoring")
except ImportError:
    try:
        from quantization import (
            QuantizationConfig,
            quantize_paligemma as quantize_paligemma_with_monitoring,
            quantize_int8 as quantize_int8_with_monitoring,
            quantize_int4 as quantize_int4_with_monitoring, 
            quantize_nf4 as quantize_nf4_with_monitoring,
            create_calibration_data
        )
        USE_ENHANCED = False
        PerformanceMetrics = None
        QuantizationPerformanceMonitor = None
        print("Using basic quantization (enhanced features not available)")
    except ImportError as e:
        print(f"Error importing quantization modules: {e}")
        sys.exit(1)


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
            print(f"Error: Image file not found: {f_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to open image {f_path}: {e}")
            sys.exit(1)

    if not images:
        raise ValueError("No valid images found for processing.")
    
    if len(images) != len(prompts):
        raise ValueError(
            f"Mismatch: Successfully loaded {len(images)} images, but have {len(prompts)} prompts."
        )

    model_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def sample_top_p_robustly(logits, top_p=0.9):
    """Robust top-p sampling with comprehensive error handling."""
    if logits.numel() == 0:
        raise ValueError("Empty logits tensor")
    
    # Handle extreme values and ensure finite values
    if not torch.isfinite(logits).all():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    
    # Clamp to reasonable range to avoid overflow/underflow
    logits = torch.clamp(logits, min=-100.0, max=100.0)
    
    # Ensure proper shape (batch_size, vocab_size)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    # Compute probabilities with numerical stability
    logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(logits_shifted, dim=-1)
    
    # Ensure valid probabilities
    probs = torch.clamp(probs, min=1e-12)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for top-p
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # Keep top token
    
    # Filter probabilities
    sorted_probs_filtered = sorted_probs.clone()
    sorted_probs_filtered[sorted_indices_to_remove] = 0.0
    
    # Normalize
    total_prob = sorted_probs_filtered.sum(dim=-1, keepdim=True)
    
    if total_prob.item() < 1e-10:
        # Fallback to greedy sampling
        selected_token = sorted_indices[..., 0:1]
    else:
        sorted_probs_filtered = sorted_probs_filtered / total_prob
        
        try:
            sample_idx = torch.multinomial(sorted_probs_filtered, num_samples=1)
            selected_token = sorted_indices.gather(-1, sample_idx)
        except Exception as e:
            print(f"Sampling failed, falling back to greedy: {e}")
            selected_token = sorted_indices[..., 0:1]
    
    return selected_token


def generate_with_quantized_model(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompts: List[str],
    image_file_paths: List[str],
    max_tokens_to_generate: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """Robust generation with quantized models - handles Flash Attention issues."""
    print(f"[GENERATE] Starting robust generation with quantized model")
    
    # Get initial inputs
    model_inputs = get_model_inputs(processor, [prompts[0]], [image_file_paths[0]], device)
    
    generated_tokens = []
    kv_cache = None  # We'll use approach without KV cache for quantized models
    
    model.eval()
    with torch.no_grad():
        try:
            # Initial forward pass with full context
            print("[GENERATE] Initial forward pass...")
            outputs = model(**model_inputs)
            
            # Sample first token
            next_token_logits = outputs["logits"][:, -1, :].float()  # Ensure float32 for stability
            
            if do_sample and top_p < 1.0:
                next_token = sample_top_p_robustly(next_token_logits / temperature, top_p)
            elif do_sample and temperature != 1.0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token.squeeze().item())
            
            # Continue generation token by token
            current_input_ids = torch.cat([model_inputs["input_ids"], next_token], dim=-1)
            
            for step in range(1, max_tokens_to_generate):
                # Check for EOS
                if next_token.item() == processor.tokenizer.eos_token_id:
                    print(f"[GENERATE] EOS token reached at step {step}")
                    break
                
                try:
                    # Create new attention mask
                    current_length = current_input_ids.shape[1]
                    attention_mask = torch.ones((1, current_length), device=device, dtype=torch.long)
                    
                    # Prepare inputs for next forward pass
                    inputs = {
                        "input_ids": current_input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": None  # Only needed for first pass
                    }
                    
                    # Forward pass
                    outputs = model(**inputs)
                    next_token_logits = outputs["logits"][:, -1, :].float()
                    
                    # Sample next token
                    if do_sample and top_p < 1.0:
                        next_token = sample_top_p_robustly(next_token_logits / temperature, top_p)
                    elif do_sample and temperature != 1.0:
                        probs = F.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    generated_tokens.append(next_token.squeeze().item())
                    current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                    
                    if step % 10 == 0:
                        print(f"[GENERATE] Generated {step} tokens...")
                    
                except Exception as e:
                    print(f"Error at generation step {step}: {e}")
                    print(f"Attempting recovery...")
                    
                    # Try approach - just use the last token
                    try:
                        recovery_inputs = {
                            "input_ids": next_token.reshape(1, 1),
                            "attention_mask": torch.ones((1, 1), device=device, dtype=torch.long),
                            "pixel_values": None
                        }
                        outputs = model(**recovery_inputs)
                        next_token_logits = outputs["logits"][:, -1, :].float()
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        generated_tokens.append(next_token.squeeze().item())
                        current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                    except Exception as e2:
                        print(f"Recovery also failed: {e2}")
                        print(f"Stopping generation at step {step}")
                        break
        
        except Exception as e:
            print(f"Critical error in initial forward pass: {e}")
            return "Error: Generation failed completely"
    
    # Decode output
    if generated_tokens:
        try:
            # Decode only the generated tokens
            final_tokens = torch.tensor(generated_tokens, device=device)
            decoded_output = processor.tokenizer.decode(final_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"Decoding error: {e}")
            # Fallback: decode each token individually and concatenate
            decoded_parts = []
            for token_id in generated_tokens:
                try:
                    part = processor.tokenizer.decode([token_id], skip_special_tokens=True)
                    decoded_parts.append(part)
                except:
                    decoded_parts.append(f"<UNK_{token_id}>")
            decoded_output = "".join(decoded_parts)
    else:
        decoded_output = "(No tokens generated)"
    
    print(f"[GENERATE] Generated {len(generated_tokens)} tokens successfully")
    return decoded_output


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(model: torch.nn.Module, name: str = "Model"):
    """Print model information."""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = get_model_size_mb(model)
    
    print(f"\n{name} Information:")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {size_mb:.2f} MB")
    
    # Check for quantized layers
    quantized_layers = 0
    total_linear_layers = 0
    
    for module in model.modules():
        if hasattr(module, '__class__'):
            class_name = module.__class__.__name__
            if 'Linear' in class_name:
                total_linear_layers += 1
                if 'Quantized' in class_name:
                    quantized_layers += 1
    
    if quantized_layers > 0:
        print(f"Quantized layers: {quantized_layers}/{total_linear_layers} ({quantized_layers/total_linear_layers*100:.1f}%)")
    print(f"Quantization: {'Enabled' if quantized_layers > 0 else 'Disabled'}")


def main(
    target_model_path: str = None,
    prompt: str = "What is in this image?",
    image_file_path: str = None,
    max_tokens_to_generate: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
    only_cpu: bool = False,
    
    # Quantization parameters
    target_quantization: str = "none",  # "none", "int8", "int4", "nf4"
    group_size: int = 128,
    
    # Enhanced monitoring options
    enable_performance_monitoring: bool = True,
    benchmark_performance: bool = False,
    
    # Options
    save_quantized: Optional[str] = None,
    use_robust_generation: bool = True,
):
    """
    Main function with robust quantized model handling.
    """
    
    # Validate required arguments
    if not target_model_path:
        print("Error: --target_model_path is required")
        sys.exit(1)
    
    if not image_file_path:
        print("Error: --image_file_path is required")
        sys.exit(1)
    
    torch.manual_seed(42)
    
    # Device setup
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)
    print(f"Target model: {target_model_path}")
    print(f"Target quantization: {target_quantization}")
    print(f"Performance monitoring: {enable_performance_monitoring}")
    print(f"Performance benchmarking: {benchmark_performance}")

    # Load base model
    print("\nLoading target model...")
    try:
        target_model, tokenizer = load_hf_model(target_model_path, device)
        # if device == "cuda":
        #     # Using bfloat16 and torch.compile for performance on CUDA-enabled GPUs
        #     target_model = torch.compile(target_model,mode="reduce-overhead")
        #     print("torch compile")

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Create processor early for potential benchmarking
    num_image_tokens = target_model.config.vision_config.num_image_tokens
    image_size = target_model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Apply quantization if requested
    metrics = None
    if target_quantization != "none":
        print(f"\nApplying {target_quantization} quantization...")
        
        try:
            if USE_ENHANCED:
                # Use enhanced quantization with monitoring
                if target_quantization == "int8":
                    target_model, metrics = quantize_int8_with_monitoring(
                        target_model, 
                        group_size=group_size,
                        device=device,
                        enable_monitoring=enable_performance_monitoring,
                        benchmark_performance=benchmark_performance,
                        processor=processor,
                        test_prompt=prompt,
                        test_image_path=image_file_path
                    )
                elif target_quantization == "int4":
                    target_model, metrics = quantize_int4_with_monitoring(
                        target_model, 
                        group_size=group_size,
                        device=device,
                        enable_monitoring=enable_performance_monitoring,
                        benchmark_performance=benchmark_performance,
                        processor=processor,
                        test_prompt=prompt,
                        test_image_path=image_file_path
                    )
                elif target_quantization == "nf4":
                    target_model, metrics = quantize_nf4_with_monitoring(
                        target_model, 
                        group_size=group_size,
                        device=device,
                        enable_monitoring=enable_performance_monitoring,
                        benchmark_performance=benchmark_performance,
                        processor=processor,
                        test_prompt=prompt,
                        test_image_path=image_file_path
                    )
                else:
                    raise ValueError(f"Unknown quantization type: {target_quantization}")
            else:
                # Use basic quantization
                if target_quantization == "int8":
                    target_model = quantize_int8_with_monitoring(target_model, group_size)
                elif target_quantization == "int4":
                    target_model = quantize_int4_with_monitoring(target_model, group_size)
                elif target_quantization == "nf4":
                    target_model = quantize_nf4_with_monitoring(target_model, group_size)
                else:
                    raise ValueError(f"Unknown quantization type: {target_quantization}")
                
            print("Quantization completed successfully!")
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Continuing with original model...")
    
    else:
        # Even without quantization, we can provide model info
        if enable_performance_monitoring and USE_ENHANCED and QuantizationPerformanceMonitor:
            monitor = QuantizationPerformanceMonitor(device)
            orig_counts, orig_memory = monitor.get_model_memory_info(target_model)
            
            print(f"\nOriginal Model Information:")
            print(f"{'='*50}")
            print(f"Total parameters: {orig_counts['total']:,}")
            print(f"Model size: {orig_memory['total_mb']:.2f} MB ({orig_memory['total_mb']/1024:.3f} GB)")

    print_model_info(target_model, "Target Model")

    # Convert to appropriate dtype if using CUDA
    if device == "cuda":
        try:
            target_model = target_model.to(torch.bfloat16)
            print("Model converted to bfloat16")
        except Exception as e:
            print(f"Warning: Failed to convert to bfloat16: {e}")

    # Save quantized model if requested
    if save_quantized:
        try:
            print(f"\nSaving quantized model to {save_quantized}")
            os.makedirs(save_quantized, exist_ok=True)
            torch.save({
                'model_state_dict': target_model.state_dict(),
                'config': target_model.config,
                'quantization': target_quantization,
                'group_size': group_size,
                'metrics': metrics.__dict__ if metrics else None
            }, os.path.join(save_quantized, "quantized_model.pt"))
            tokenizer.save_pretrained(save_quantized)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Failed to save model: {e}")

    # Run generation
    print(f"\nRunning generation...")
    print(f"Prompt: '{prompt}'")
    print(f"Image: {image_file_path}")
    print(f"Using robust generation: {use_robust_generation}")

    start_time = time.time()
    
    try:
        if use_robust_generation:
            generated_text = generate_with_quantized_model(
                model=target_model,
                processor=processor,
                device=device,
                prompts=[prompt],
                image_file_paths=[image_file_path],
                max_tokens_to_generate=max_tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        else:
            print("Note: Using fallback generation (may have issues with quantized models)")
            # Fallback to your original generation function here if needed
            generated_text = "Fallback generation not implemented"
        
        generation_time = time.time() - start_time
        tokens_generated = len(generated_text.split())  # Rough token count
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        print("\n" + "="*80)
        print("GENERATION RESULTS")
        print("="*80)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Generation time: {generation_time:.3f}s")
        print(f"Approximate tokens/second: {tokens_per_second:.2f}")
        print(f"Quantization: {target_quantization}")
        
        # If we have metrics and didn't benchmark during quantization, update with actual performance
        if USE_ENHANCED and metrics and not benchmark_performance and tokens_per_second > 0:
            print(f"\nUpdating performance metrics with actual generation speed...")
            metrics.tokens_per_second = tokens_per_second
            metrics.actual_bandwidth_gbps = metrics.model_size_gb * tokens_per_second
            
            monitor = QuantizationPerformanceMonitor(device)
            metrics.model_bandwidth_utilization = (
                (metrics.actual_bandwidth_gbps / monitor.gpu_bandwidth_gbps) * 100
            )
            
            print(f"Model Bandwidth Utilization: {metrics.model_bandwidth_utilization:.1f}%")
        
        print("="*80)
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        fire.Fire(main)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)