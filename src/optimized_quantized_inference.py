
"""
Optimized quantized inference script with performance enhancements.
"""

from PIL import Image
import torch
import torch.nn.functional as F
import fire
from typing import List, Optional, Tuple
import time
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing_paligemma import PaliGemmaProcessor
from gemma_flash import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

# Import optimized quantization functions
try:
    from optimized_quantization_utils import (
        OptimizedQuantizationConfig,
        quantize_with_optimizations,
        quantize_int8_optimized,
        quantize_int4_optimized, 
        quantize_nf4_optimized,
        PerformanceMetrics,
        OptimizedPerformanceMonitor
    )
    USE_OPTIMIZED = True
    print("Using optimized quantization with performance enhancements")
except ImportError:
    # Fallback to original quantization
    try:
        from quantization_utils import (
            QuantizationConfig as OptimizedQuantizationConfig,
            quantize_paligemma_with_monitoring as quantize_with_optimizations,
            quantize_int8_with_monitoring as quantize_int8_optimized,
            quantize_int4_with_monitoring as quantize_int4_optimized, 
            quantize_nf4_with_monitoring as quantize_nf4_optimized,
            PerformanceMetrics,
            QuantizationPerformanceMonitor as OptimizedPerformanceMonitor
        )
        USE_OPTIMIZED = False
        print("Using fallback quantization (optimized features not available)")
    except ImportError as e:
        print(f"Error importing quantization modules: {e}")
        sys.exit(1)


def move_inputs_to_device(model_inputs: dict, device: str):
    """Move model inputs to the specified device."""
    return {k: v.to(device) for k, v in model_inputs.items()}


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


@torch.compile(mode="reduce-overhead", fullgraph=True)
def optimized_sample_top_p(logits, top_p=0.9, temperature=1.0):
    """Optimized top-p sampling with torch.compile acceleration."""
    if temperature != 1.0:
        logits = logits / temperature
    
    # Numerical stability
    logits = torch.clamp(logits, min=-50.0, max=50.0)
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for top-p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Filter probabilities
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Sample
    try:
        sample_idx = torch.multinomial(sorted_probs, num_samples=1)
        selected_token = sorted_indices.gather(-1, sample_idx)
    except:
        # Fallback to greedy
        selected_token = sorted_indices[..., 0:1]
    
    return selected_token


def optimized_generate_with_quantized_model(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompts: List[str],
    image_file_paths: List[str],
    max_tokens_to_generate: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_torch_compile: bool = True
):
    """Optimized generation with performance enhancements."""
    print(f"[GENERATE] Starting optimized generation")
    print(f"[GENERATE] torch.compile enabled: {use_torch_compile}")
    
    # Get initial inputs
    model_inputs = get_model_inputs(processor, [prompts[0]], [image_file_paths[0]], device)
    
    generated_tokens = []
    
    # Enable optimizations
    model.eval()
    
    # Optional: Enable attention optimizations
    if hasattr(model, 'config'):
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True
        if hasattr(model.config, 'torch_dtype'):
            model.config.torch_dtype = torch.bfloat16
    
    with torch.no_grad():
        # Use torch.inference_mode for better performance
        with torch.inference_mode():
            try:
                # Initial forward pass
                print("[GENERATE] Initial forward pass...")
                
                # Warmup for torch.compile
                if use_torch_compile:
                    try:
                        _ = model(**model_inputs)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        print("[GENERATE] Warmup completed")
                    except:
                        print("[GENERATE] Warmup failed, continuing...")
                
                # Actual generation
                outputs = model(**model_inputs)
                next_token_logits = outputs["logits"][:, -1, :].float()
                
                # Optimized sampling
                if do_sample:
                    if use_torch_compile:
                        next_token = optimized_sample_top_p(next_token_logits, top_p, temperature)
                    else:
                        probs = F.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_tokens.append(next_token.squeeze().item())
                
                # Continue generation with optimized loop
                current_input_ids = torch.cat([model_inputs["input_ids"], next_token], dim=-1)
                
                # Pre-allocate attention mask for efficiency
                max_length = model_inputs["input_ids"].shape[1] + max_tokens_to_generate
                attention_mask = torch.ones((1, max_length), device=device, dtype=torch.long)
                attention_mask[:, model_inputs["input_ids"].shape[1]:] = 0
                
                for step in range(1, max_tokens_to_generate):
                    # Check for EOS
                    if next_token.item() == processor.tokenizer.eos_token_id:
                        print(f"[GENERATE] EOS token reached at step {step}")
                        break
                    
                    try:
                        # Update attention mask
                        current_length = current_input_ids.shape[1]
                        attention_mask[:, :current_length] = 1
                        
                        # Prepare inputs - only pass what's needed
                        inputs = {
                            "input_ids": current_input_ids,
                            "attention_mask": attention_mask[:, :current_length],
                            "pixel_values": None  # Only needed for first pass
                        }
                        
                        # Forward pass with potential caching
                        outputs = model(**inputs)
                        next_token_logits = outputs["logits"][:, -1, :].float()
                        
                        # Optimized sampling
                        if do_sample:
                            if use_torch_compile:
                                next_token = optimized_sample_top_p(next_token_logits, top_p, temperature)
                            else:
                                probs = F.softmax(next_token_logits / temperature, dim=-1)
                                next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        generated_tokens.append(next_token.squeeze().item())
                        current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                        
                        # Progress reporting
                        if step % 10 == 0:
                            print(f"[GENERATE] Generated {step}/{max_tokens_to_generate} tokens...")
                        
                    except Exception as e:
                        print(f"Error at step {step}: {e}")
                        break
                
            except Exception as e:
                print(f"Critical error in generation: {e}")
                return "Error: Generation failed"
    
    # Decode output
    if generated_tokens:
        try:
            final_tokens = torch.tensor(generated_tokens, device=device)
            decoded_output = processor.tokenizer.decode(final_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"Decoding error: {e}")
            decoded_output = "Error: Failed to decode tokens"
    else:
        decoded_output = "(No tokens generated)"
    
    print(f"[GENERATE] Generated {len(generated_tokens)} tokens successfully")
    return decoded_output


def print_optimized_model_info(model: torch.nn.Module, name: str = "Model"):
    """Print detailed model information with optimization status."""
    total_params = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    print(f"\n{name} Information:")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Memory breakdown:")
    print(f"  Parameters: {param_size/(1024*1024):.2f} MB")
    print(f"  Buffers: {buffer_size/(1024*1024):.2f} MB")
    
    # Check for quantized and compiled layers
    quantized_layers = 0
    compiled_layers = 0
    total_linear_layers = 0
    
    for module in model.modules():
        if hasattr(module, '__class__'):
            class_name = module.__class__.__name__
            if 'Linear' in class_name:
                total_linear_layers += 1
                if 'Quantized' in class_name or 'Optimized' in class_name:
                    quantized_layers += 1
            if hasattr(module, '_compiled'):
                compiled_layers += 1
    
    if quantized_layers > 0:
        print(f"Quantized layers: {quantized_layers}/{total_linear_layers} ({quantized_layers/total_linear_layers*100:.1f}%)")
    
    print(f"Quantization: {'Enabled' if quantized_layers > 0 else 'Disabled'}")
    print(f"Compiled layers: {compiled_layers}")


def main(
    target_model_path: str = None,
    prompt: str = "What is in this image?",
    image_file_path: str = None,
    max_tokens_to_generate: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
    only_cpu: bool = False,
    
    # Optimized quantization parameters
    target_quantization: str = "none",  # "none", "int8", "int4", "nf4"
    group_size: int = 64,  # Smaller for better performance
    
    # Performance options
    enable_performance_monitoring: bool = True,
    benchmark_performance: bool = False,
    use_torch_compile: bool = True,
    enable_optimizations: bool = True,
    
    # Other options
    save_quantized: Optional[str] = None,
):
    """
    Optimized main function with performance enhancements.
    """
    
    # Validate required arguments
    if not target_model_path:
        print("Error: --target_model_path is required")
        sys.exit(1)
    
    if not image_file_path:
        print("Error: --image_file_path is required")
        sys.exit(1)
    
    torch.manual_seed(42)
    
    # Device setup with optimization
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
            # Enable optimizations for CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use:", device)
    print(f"Target model: {target_model_path}")
    print(f"Target quantization: {target_quantization}")
    print(f"Optimizations enabled: {enable_optimizations}")
    print(f"torch.compile enabled: {use_torch_compile}")

    # Load model
    print("\nLoading target model...")
    try:
        target_model, tokenizer = load_hf_model(target_model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Create processor
    num_image_tokens = target_model.config.vision_config.num_image_tokens
    image_size = target_model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Apply quantization with optimizations
    metrics = None
    if target_quantization != "none":
        print(f"\nApplying optimized {target_quantization} quantization...")
        
        try:
            if target_quantization == "int8":
                target_model, metrics = quantize_int8_optimized(
                    target_model, 
                    group_size=group_size,
                    device=device,
                    enable_performance_monitoring=enable_performance_monitoring,
                    benchmark_performance=benchmark_performance,
                    processor=processor,
                    test_prompt=prompt,
                    test_image_path=image_file_path
                )
            elif target_quantization == "int4":
                target_model, metrics = quantize_int4_optimized(
                    target_model, 
                    group_size=group_size,
                    device=device,
                    enable_performance_monitoring=enable_performance_monitoring,
                    benchmark_performance=benchmark_performance,
                    processor=processor,
                    test_prompt=prompt,
                    test_image_path=image_file_path
                )
            elif target_quantization == "nf4":
                target_model, metrics = quantize_nf4_optimized(
                    target_model, 
                    group_size=group_size,
                    device=device,
                    enable_performance_monitoring=enable_performance_monitoring,
                    benchmark_performance=benchmark_performance,
                    processor=processor,
                    test_prompt=prompt,
                    test_image_path=image_file_path
                )
            else:
                raise ValueError(f"Unknown quantization type: {target_quantization}")
                
            print("Optimized quantization completed!")
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Continuing with original model...")



    # Convert to appropriate dtype
    if device == "cuda":
        try:
            target_model = target_model.to(torch.bfloat16)
            print("Model converted to bfloat16")
        except Exception as e:
            print(f"Warning: Failed to convert to bfloat16: {e}")

    # Save model if requested
    if save_quantized:
        try:
            print(f"\nSaving optimized model to {save_quantized}")
            os.makedirs(save_quantized, exist_ok=True)
            torch.save({
                'model_state_dict': target_model.state_dict(),
                'config': target_model.config,
                'quantization': target_quantization,
                'group_size': group_size,
                'optimizations': {
                    'torch_compile': use_torch_compile,
                    'optimizations_enabled': enable_optimizations
                },
                'metrics': metrics.__dict__ if metrics else None
            }, os.path.join(save_quantized, "optimized_model.pt"))
            tokenizer.save_pretrained(save_quantized)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Failed to save model: {e}")

    # Run optimized generation
    print(f"\nRunning optimized generation...")
    print(f"Prompt: '{prompt}'")
    print(f"Image: {image_file_path}")

    start_time = time.time()
    
    try:
        generated_text = optimized_generate_with_quantized_model(
            model=target_model,
            processor=processor,
            device=device,
            prompts=[prompt],
            image_file_paths=[image_file_path],
            max_tokens_to_generate=max_tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_torch_compile=use_torch_compile
        )
        
        generation_time = time.time() - start_time
        tokens_generated = len(generated_text.split())
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        print("\n" + "="*80)
        print("OPTIMIZED GENERATION RESULTS")
        print("="*80)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Generation time: {generation_time:.3f}s")
        print(f"Tokens/second: {tokens_per_second:.2f}")
        print(f"Quantization: {target_quantization}")
        print(f"Optimizations: {'Enabled' if enable_optimizations else 'Disabled'}")
        
        # Update metrics with actual performance
        if USE_OPTIMIZED and metrics and not benchmark_performance:
            print(f"\nUpdating performance metrics...")
            metrics.tokens_per_second = tokens_per_second
            metrics.actual_bandwidth_gbps = metrics.model_size_gb * tokens_per_second
            
            monitor = OptimizedPerformanceMonitor(device)
            metrics.model_bandwidth_utilization = (
                (metrics.actual_bandwidth_gbps / monitor.gpu_bandwidth_gbps) * 100
            )
            
            print(f"Updated MBU: {metrics.model_bandwidth_utilization:.2f}%")
            
            # Print improvement suggestions
            if metrics.model_bandwidth_utilization < 10:
                print("\nPERFORMANCE SUGGESTIONS:")
                print("- Model is still compute-bound")
                print("- Consider enabling Flash Attention 2.0")
                print("- Try different torch.compile modes")
                print("- Profile attention computation overhead")
        
        print("="*80)
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
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

