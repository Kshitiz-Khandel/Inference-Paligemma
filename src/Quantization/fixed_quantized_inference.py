"""
Fixed quantized inference script with proper imports and error handling.
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

# Import the quantization functions - try enhanced first, fallback to basic
try:
    from enhanced_quantization import (
        QuantizationConfig,
        quantize_paligemma_enhanced,
        quantize_int8_enhanced,
        quantize_int4_enhanced, 
        quantize_nf4_enhanced,
        PerformanceMetrics,
        QuantizationPerformanceMonitor
    )
    USE_ENHANCED = True
    print("Using enhanced quantization with performance monitoring")
except ImportError:
    try:
        from fixed_quantization import (
            QuantizationConfig,
            quantize_paligemma_simple as quantize_paligemma_enhanced,
            quantize_int8_simple as quantize_int8_enhanced,
            quantize_int4_simple as quantize_int4_enhanced, 
            quantize_nf4_simple as quantize_nf4_enhanced,
            create_fixed_calibration_data
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


def _sample_top_p(logits, top_p=0.9):
    """Simple top-p sampling."""
    if logits.numel() == 0:
        raise ValueError("Empty logits tensor")
    
    # Handle extreme values
    if not torch.isfinite(logits).all():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    
    # Clamp to reasonable range
    logits = torch.clamp(logits, min=-100.0, max=100.0)
    
    # Compute probabilities
    logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(logits_shifted, dim=-1)
    
    # Ensure valid probabilities
    probs = torch.clamp(probs, min=1e-12)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
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
        # Fallback to greedy
        selected_token = sorted_indices[..., 0:1]
    else:
        sorted_probs_filtered = sorted_probs_filtered / total_prob
        
        try:
            sample_idx = torch.multinomial(sorted_probs_filtered, num_samples=1)
            selected_token = sorted_indices.gather(-1, sample_idx)
        except:
            selected_token = sorted_indices[..., 0:1]
    
    # Ensure proper shape
    if selected_token.dim() == 1:
        selected_token = selected_token.unsqueeze(0)
    elif selected_token.dim() == 0:
        selected_token = selected_token.unsqueeze(0).unsqueeze(0)
    
    return selected_token


def simple_generate_with_quantized_model(
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
    """Simple generation that should work reliably with quantized models."""
    print(f"[GENERATE] Starting simple generation")
    
    # Get initial inputs
    model_inputs = get_model_inputs(processor, [prompts[0]], [image_file_paths[0]], device)
    
    generated_tokens = []
    
    with torch.no_grad():
        # Initial forward pass
        try:
            outputs = model(**model_inputs)
        except Exception as e:
            print(f"Error in initial forward pass: {e}")
            return "Error: Initial forward pass failed"
        
        # Sample first token
        next_token_logits = outputs["logits"][:, -1, :]
        if do_sample and top_p < 1.0:
            next_token = _sample_top_p(next_token_logits / temperature, top_p)
        elif do_sample:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        generated_tokens.append(next_token.squeeze().item())
        
        # Continue generation (simplified approach)
        for step in range(1, max_tokens_to_generate):
            if next_token.item() == processor.tokenizer.eos_token_id:
                break
            
            try:
                # Prepare input for next token (just the last generated token)
                input_ids = next_token.reshape(1, 1)
                
                # Create attention mask - simplified approach
                seq_len = model_inputs["input_ids"].shape[1] + step
                attention_mask = torch.ones((1, seq_len), device=device)
                
                # Forward pass
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": None  # No new images needed
                }
                
                outputs = model(**inputs)
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Sample next token
                if do_sample and top_p < 1.0:
                    next_token = _sample_top_p(next_token_logits / temperature, top_p)
                elif do_sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_tokens.append(next_token.squeeze().item())
                
            except Exception as e:
                print(f"Error at step {step}: {e}")
                print(f"Stopping generation early")
                break
    
    # Decode output
    if generated_tokens:
        final_tokens = torch.tensor(generated_tokens, device=device)
        decoded_output = processor.tokenizer.decode(final_tokens, skip_special_tokens=True)
    else:
        decoded_output = "(No tokens generated)"
    
    print(f"[GENERATE] Generated {len(generated_tokens)} tokens")
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
):
    """
    Enhanced main function with comprehensive performance monitoring.
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
                    target_model, metrics = quantize_int8_enhanced(
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
                    target_model, metrics = quantize_int4_enhanced(
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
                    target_model, metrics = quantize_nf4_enhanced(
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
                    target_model = quantize_int8_enhanced(target_model, group_size)
                elif target_quantization == "int4":
                    target_model = quantize_int4_enhanced(target_model, group_size)
                elif target_quantization == "nf4":
                    target_model = quantize_nf4_enhanced(target_model, group_size)
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

    start_time = time.time()
    
    try:
        generated_text = simple_generate_with_quantized_model(
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













# """
# Fixed quantized inference script that addresses the tensor indexing issues.
# """

# from PIL import Image
# import torch
# import torch.nn.functional as F
# import fire
# from typing import List, Optional, Tuple
# import time
# import os

# from processing_paligemma import PaliGemmaProcessor
# from gemma_flash import KVCache, PaliGemmaForConditionalGeneration
# from utils import load_hf_model
# from fixed_quantization import (
#     QuantizationConfig,
#     quantize_paligemma_simple,
#     quantize_int8_simple,
#     quantize_int4_simple, 
#     quantize_nf4_simple,
#     create_fixed_calibration_data
# )


# def move_inputs_to_device(model_inputs: dict, device: str):
#     """Move model inputs to the specified device."""
#     model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
#     return model_inputs


# def get_model_inputs(
#     processor: PaliGemmaProcessor, 
#     prompts: List[str], 
#     image_file_paths: List[str], 
#     device: str
# ):
#     """Load images and prepare model inputs."""
#     images = []
#     for f_path in image_file_paths:
#         try:
#             img = Image.open(f_path).convert("RGB")
#             images.append(img)
#         except FileNotFoundError:
#             print(f"Warning: Image file not found: {f_path}. Skipping this image.")
#         except Exception as e:
#             print(f"Warning: Failed to open image {f_path}: {e}. Skipping this image.")

#     if not images:
#         raise ValueError("No valid images found for processing.")
    
#     if len(images) != len(prompts):
        
#         raise ValueError(
#             f"Mismatch: Successfully loaded {len(images)} images, but have {len(prompts)} prompts. "
#             "Ensure all image files exist and are accessible."
# )

#     model_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
#     model_inputs = move_inputs_to_device(model_inputs, device)
#     return model_inputs


# def _sample_top_p(logits, top_p=0.9):
#     """Simple top-p sampling."""
#     if logits.numel() == 0:
#         raise ValueError("Empty logits tensor")
    
#     # Handle extreme values
#     if not torch.isfinite(logits).all():
#         logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    
#     # Clamp to reasonable range
#     logits = torch.clamp(logits, min=-100.0, max=100.0)
    
#     # Compute probabilities
#     logits_shifted = logits - logits.max(dim=-1, keepdim=True).values
#     probs = F.softmax(logits_shifted, dim=-1)
    
#     # Ensure valid probabilities
#     probs = torch.clamp(probs, min=1e-12)
#     probs = probs / probs.sum(dim=-1, keepdim=True)
    
#     # Sort probabilities
#     sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
#     # Compute cumulative probabilities
#     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
#     # Create mask for top-p
#     sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
#     sorted_indices_to_remove[..., 0] = False  # Keep top token
    
#     # Filter probabilities
#     sorted_probs_filtered = sorted_probs.clone()
#     sorted_probs_filtered[sorted_indices_to_remove] = 0.0
    
#     # Normalize
#     total_prob = sorted_probs_filtered.sum(dim=-1, keepdim=True)
    
#     if total_prob.item() < 1e-10:
#         # Fallback to greedy
#         selected_token = sorted_indices[..., 0:1]
#     else:
#         sorted_probs_filtered = sorted_probs_filtered / total_prob
        
#         try:
#             sample_idx = torch.multinomial(sorted_probs_filtered, num_samples=1)
#             selected_token = sorted_indices.gather(-1, sample_idx)
#         except:
#             selected_token = sorted_indices[..., 0:1]
    
#     # Ensure proper shape
#     if selected_token.dim() == 1:
#         selected_token = selected_token.unsqueeze(0)
#     elif selected_token.dim() == 0:
#         selected_token = selected_token.unsqueeze(0).unsqueeze(0)
    
#     return selected_token


# def simple_generate_with_quantized_model(
#     model: PaliGemmaForConditionalGeneration,
#     processor: PaliGemmaProcessor,
#     device: str,
#     prompts: List[str],
#     image_file_paths: List[str],
#     max_tokens_to_generate: int,
#     temperature: float = 0.8,
#     top_p: float = 0.9,
#     do_sample: bool = True,
# ):
#     """Simple generation that should work reliably with quantized models."""
#     print(f"[GENERATE] Starting simple generation")
    
#     # Get initial inputs
#     model_inputs = get_model_inputs(processor, [prompts[0]], [image_file_paths[0]], device)
    
#     generated_tokens = []
    
#     with torch.no_grad():
#         # Initial forward pass
#         try:
#             outputs = model(**model_inputs)
#         except Exception as e:
#             print(f"Error in initial forward pass: {e}")
#             return "Error: Initial forward pass failed"
        
#         # Sample first token
#         next_token_logits = outputs["logits"][:, -1, :]
#         if do_sample and top_p < 1.0:
#             next_token = _sample_top_p(next_token_logits / temperature, top_p)
#         elif do_sample:
#             probs = F.softmax(next_token_logits / temperature, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#         else:
#             next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
#         generated_tokens.append(next_token.squeeze().item())
        
#         # Continue generation (simplified approach)
#         for step in range(1, max_tokens_to_generate):
#             if next_token.item() == processor.tokenizer.eos_token_id:
#                 break
            
#             try:
#                 # Prepare input for next token (just the last generated token)
#                 input_ids = next_token.reshape(1, 1)
                
#                 # Create attention mask - simplified approach
#                 seq_len = model_inputs["input_ids"].shape[1] + step
#                 attention_mask = torch.ones((1, seq_len), device=device)
                
#                 # Forward pass
#                 inputs = {
#                     "input_ids": input_ids,
#                     "attention_mask": attention_mask,
#                     "pixel_values": None  # No new images needed
#                 }
                
#                 outputs = model(**inputs)
#                 next_token_logits = outputs["logits"][:, -1, :]
                
#                 # Sample next token
#                 if do_sample and top_p < 1.0:
#                     next_token = _sample_top_p(next_token_logits / temperature, top_p)
#                 elif do_sample:
#                     probs = F.softmax(next_token_logits / temperature, dim=-1)
#                     next_token = torch.multinomial(probs, num_samples=1)
#                 else:
#                     next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
#                 generated_tokens.append(next_token.squeeze().item())
                
#             except Exception as e:
#                 print(f"Error at step {step}: {e}")
#                 print(f"Stopping generation early")
#                 break
    
#     # Decode output
#     if generated_tokens:
#         final_tokens = torch.tensor(generated_tokens, device=device)
#         decoded_output = processor.tokenizer.decode(final_tokens, skip_special_tokens=True)
#     else:
#         decoded_output = "(No tokens generated)"
    
#     print(f"[GENERATE] Generated {len(generated_tokens)} tokens")
#     return decoded_output


# def get_model_size_mb(model: torch.nn.Module) -> float:
#     """Calculate model size in MB."""
#     param_size = sum(p.numel() * p.element_size() for p in model.parameters())
#     buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
#     size_mb = (param_size + buffer_size) / 1024 / 1024
#     return size_mb


# def print_model_info(model: torch.nn.Module, name: str = "Model"):
#     """Print model information."""
#     total_params = sum(p.numel() for p in model.parameters())
#     size_mb = get_model_size_mb(model)
    
#     print(f"\n{name} Information:")
#     print(f"{'='*50}")
#     print(f"Total parameters: {total_params:,}")
#     print(f"Model size: {size_mb:.2f} MB")
    
#     # Check for quantized layers
#     quantized_layers = 0
#     total_linear_layers = 0
    
#     for module in model.modules():
#         if hasattr(module, '__class__'):
#             class_name = module.__class__.__name__
#             if 'Linear' in class_name:
#                 total_linear_layers += 1
#                 if 'Quantized' in class_name:
#                     quantized_layers += 1
    
#     if quantized_layers > 0:
#         print(f"Quantized layers: {quantized_layers}/{total_linear_layers} ({quantized_layers/total_linear_layers*100:.1f}%)")
#     print(f"Quantization: {'Enabled' if quantized_layers > 0 else 'Disabled'}")


# def main(
#     target_model_path: str = None,
#     prompt: str = "What is in this image?",
#     image_file_path: str = None,
#     max_tokens_to_generate: int = 50,
#     temperature: float = 0.8,
#     top_p: float = 0.9,
#     do_sample: bool = True,
#     only_cpu: bool = False,
    
#     # Quantization parameters
#     target_quantization: str = "none",  # "none", "int8", "int4", "nf4"
#     group_size: int = 128,
    
#     # Options
#     save_quantized: Optional[str] = None,
# ):
#     """
#     Fixed main function for quantized inference.
#     """
#     torch.manual_seed(42)
    
#     # Device setup
#     device = "cpu"
#     if not only_cpu:
#         if torch.cuda.is_available():
#             device = "cuda"
#         elif torch.backends.mps.is_available():
#             device = "mps"

#     print("Device in use: ", device)
#     print(f"Target model: {target_model_path}")
#     print(f"Target quantization: {target_quantization}")

#     # Load base model
#     print("\nLoading target model...")
#     target_model, tokenizer = load_hf_model(target_model_path, device)

#     # Apply quantization if requested
#     if target_quantization != "none":
#         print(f"\nApplying {target_quantization} quantization...")
        
#         try:
#             if target_quantization == "int8":
#                 target_model = quantize_int8_simple(target_model, group_size)
#             elif target_quantization == "int4":
#                 target_model = quantize_int4_simple(target_model, group_size)
#             elif target_quantization == "nf4":
#                 target_model = quantize_nf4_simple(target_model, group_size)
#             else:
#                 raise ValueError(f"Unknown quantization type: {target_quantization}")
                
#             print("Quantization completed successfully!")
            
#         except Exception as e:
#             print(f"Quantization failed: {e}")
#             print("Continuing with original model...")

#     print_model_info(target_model, "Target Model")

#     # Convert to appropriate dtype if using CUDA
#     if device == "cuda":
#         try:
#             target_model = target_model.to(torch.bfloat16)
#             print("Model converted to bfloat16")
#         except Exception as e:
#             print(f"Warning: Failed to convert to bfloat16: {e}")

#     # Create processor
#     num_image_tokens = target_model.config.vision_config.num_image_tokens
#     image_size = target_model.config.vision_config.image_size
#     processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

#     # Save quantized model if requested
#     if save_quantized:
#         try:
#             print(f"\nSaving quantized model to {save_quantized}")
#             os.makedirs(save_quantized, exist_ok=True)
#             torch.save({
#                 'model_state_dict': target_model.state_dict(),
#                 'config': target_model.config,
#                 'quantization': target_quantization,
#                 'group_size': group_size
#             }, os.path.join(save_quantized, "quantized_model.pt"))
#             tokenizer.save_pretrained(save_quantized)
#             print("Model saved successfully!")
#         except Exception as e:
#             print(f"Failed to save model: {e}")

#     # Run generation
#     print(f"\nRunning generation...")
#     print(f"Prompt: '{prompt}'")
#     print(f"Image: {image_file_path}")

#     start_time = time.time()
    
#     try:
#         generated_text = simple_generate_with_quantized_model(
#             model=target_model,
#             processor=processor,
#             device=device,
#             prompts=[prompt],
#             image_file_paths=[image_file_path],
#             max_tokens_to_generate=max_tokens_to_generate,
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=do_sample,
#         )
        
#         generation_time = time.time() - start_time
        
#         print("\n" + "="*80)
#         print("GENERATION RESULTS")
#         print("="*80)
#         print(f"Prompt: {prompt}")
#         print(f"Generated: {generated_text}")
#         print(f"Generation time: {generation_time:.3f}s")
#         print(f"Quantization: {target_quantization}")
#         print("="*80)
        
#     except Exception as e:
#         print(f"Generation failed: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     fire.Fire(main)