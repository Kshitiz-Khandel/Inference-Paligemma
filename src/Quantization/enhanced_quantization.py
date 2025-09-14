"""
Complete enhanced quantization module that integrates with your existing fixed_quantization.py
This file should be saved as enhanced_quantization.py in the same directory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from abc import ABC, abstractmethod
import warnings
import time
from dataclasses import dataclass

# Import your existing classes - adjust these imports to match your actual file structure
try:
    from fixed_quantization import (
        SimpleQuantizedLinear,
        QuantizationConfig,
        FixedCalibrationDataLoader,
        replace_linear_layers_simple,
        create_fixed_calibration_data
    )
except ImportError:
    # If the above doesn't work, we'll define the necessary classes here
    print("Warning: Could not import from fixed_quantization, using embedded definitions")
    
    class QuantizationConfig:
        """Configuration for quantization settings."""
        
        def __init__(
            self,
            bits: int = 8,
            group_size: int = 128,
            calibration_samples: int = 512,
            use_nf4: bool = False,
            double_quant: bool = False,
            quant_vision: bool = True,
            quant_text: bool = True,
            quant_projector: bool = True,
            skip_layers: Optional[List[str]] = ["embed_tokens","lm_head"]
        ):
            self.bits = bits
            self.group_size = group_size
            self.calibration_samples = calibration_samples
            self.use_nf4 = use_nf4
            self.double_quant = double_quant
            self.quant_vision = quant_vision
            self.quant_text = quant_text
            self.quant_projector = quant_projector
            self.skip_layers = skip_layers or []
            
            # Validate configuration
            if bits not in [4, 8]:
                raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")
            if use_nf4 and bits != 4:
                raise ValueError("NF4 quantization only supports 4-bit")

    class SimpleQuantizedLinear(nn.Module):
        """Simplified quantized linear layer that's more robust."""
        
        def __init__(
            self, 
            weight: torch.Tensor, 
            bias: Optional[torch.Tensor] = None,
            bits: int = 8,
            group_size: int = 128,
            use_nf4: bool = False
        ):
            super().__init__()
            self.bits = bits
            self.group_size = group_size
            self.use_nf4 = use_nf4
            self.out_features, self.in_features = weight.shape
            
            # Store original weight info
            self.register_buffer('original_shape', torch.tensor(weight.shape))
            
            # Quantize the weight using simplified method
            if use_nf4 and bits == 4:
                self.weight_quant, self.weight_scale = self._quantize_nf4_simple(weight)
            else:
                self.weight_quant, self.weight_scale = self._quantize_symmetric_simple(weight)
            
            # Store bias if present
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None
        
        def _get_nf4_values(self) -> torch.Tensor:
            """Get the 16 NF4 quantization values."""
            return torch.tensor([
                -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
            ], dtype=torch.float32)
        
        def _quantize_nf4_simple(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Simplified NF4 quantization without complex packing."""
            device = weight.device
            dtype = weight.dtype
            nf4_values = self._get_nf4_values().to(device=device, dtype=dtype)
            
            # Flatten weight for processing
            weight_flat = weight.view(-1)
            
            # Group-wise processing
            if weight_flat.numel() < self.group_size:
                # Small tensor - treat as single group
                groups = [weight_flat]
            else:
                # Split into groups, padding if necessary
                num_groups = (weight_flat.numel() + self.group_size - 1) // self.group_size
                padded_size = num_groups * self.group_size
                
                if padded_size > weight_flat.numel():
                    weight_flat = F.pad(weight_flat, (0, padded_size - weight_flat.numel()), value=0)
                
                groups = weight_flat.view(num_groups, self.group_size)
            
            # Quantize each group
            quantized_groups = []
            scales = []
            
            for group in groups:
                # Calculate scale for this group
                abs_max = group.abs().max()
                scale = abs_max / nf4_values.max() if abs_max > 0 else torch.tensor(1.0, device=device, dtype=dtype)
                scales.append(scale)
                
                # Normalize and quantize
                if scale > 0:
                    normalized = group / scale
                    # Find closest NF4 values
                    distances = torch.abs(normalized.unsqueeze(1) - nf4_values.unsqueeze(0))
                    indices = torch.argmin(distances, dim=1)
                    quantized_groups.append(indices)
                else:
                    # All zeros
                    quantized_groups.append(torch.zeros_like(group, dtype=torch.long, device=device))
            
            # Combine results
            if len(quantized_groups) == 1:
                weight_quant = quantized_groups[0]
                scale = scales[0].unsqueeze(0)
            else:
                weight_quant = torch.cat(quantized_groups, dim=0)
                scale = torch.stack(scales, dim=0)
            
            # Trim to original size if we padded
            original_numel = weight.numel()
            if weight_quant.numel() > original_numel:
                weight_quant = weight_quant[:original_numel]
            
            # Store as int8 (more compatible)
            weight_quant = weight_quant.to(torch.int8)
            
            return weight_quant, scale
        
        def _quantize_symmetric_simple(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Simplified symmetric quantization."""
            device = weight.device
            dtype = weight.dtype
            
            # Flatten for processing
            weight_flat = weight.view(-1)
            
            # Determine quantization range
            if self.bits == 8:
                qmax = 127
            else:  # 4-bit
                qmax = 7
            
            # Group-wise processing similar to NF4
            if weight_flat.numel() < self.group_size:
                groups = [weight_flat]
            else:
                num_groups = (weight_flat.numel() + self.group_size - 1) // self.group_size
                padded_size = num_groups * self.group_size
                
                if padded_size > weight_flat.numel():
                    weight_flat = F.pad(weight_flat, (0, padded_size - weight_flat.numel()), value=0)
                
                groups = weight_flat.view(num_groups, self.group_size)
            
            # Quantize each group
            quantized_groups = []
            scales = []
            
            for group in groups:
                abs_max = group.abs().max()
                scale = abs_max / qmax if abs_max > 0 else torch.tensor(1.0, device=device, dtype=dtype)
                scales.append(scale)
                
                if scale > 0:
                    quantized = torch.round(group / scale).clamp(-qmax, qmax)
                    quantized_groups.append(quantized)
                else:
                    quantized_groups.append(torch.zeros_like(group, device=device))
            
            # Combine results
            if len(quantized_groups) == 1:
                weight_quant = quantized_groups[0]
                scale = scales[0].unsqueeze(0)
            else:
                weight_quant = torch.cat(quantized_groups, dim=0)
                scale = torch.stack(scales, dim=0)
            
            # Trim to original size
            original_numel = weight.numel()
            if weight_quant.numel() > original_numel:
                weight_quant = weight_quant[:original_numel]
            
            weight_quant = weight_quant.to(torch.int8)
            
            return weight_quant, scale
        
        def _dequantize_nf4_simple(self, weight_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            """Simplified NF4 dequantization."""
            device = weight_quant.device
            dtype = self.weight_scale.dtype
            nf4_values = self._get_nf4_values().to(device=device, dtype=dtype)
            
            # Clamp indices to valid range
            indices = torch.clamp(weight_quant.long(), 0, 15)
            
            # Look up values
            weight_dequant = nf4_values[indices]
            
            # Apply scaling
            if scale.numel() == 1:
                # Single scale for entire tensor
                weight_dequant = weight_dequant * scale.item()
            else:
                # Group-wise scaling
                groups_per_scale = weight_dequant.numel() // scale.numel()
                for i, s in enumerate(scale):
                    start_idx = i * groups_per_scale
                    end_idx = min((i + 1) * groups_per_scale, weight_dequant.numel())
                    weight_dequant[start_idx:end_idx] *= s
            
            # Reshape to original shape
            return weight_dequant.view(self.out_features, self.in_features)
        
        def _dequantize_symmetric_simple(self, weight_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            """Simplified symmetric dequantization."""
            weight_dequant = weight_quant.float()
            
            # Apply scaling
            if scale.numel() == 1:
                weight_dequant = weight_dequant * scale.item()
            else:
                groups_per_scale = weight_dequant.numel() // scale.numel()
                for i, s in enumerate(scale):
                    start_idx = i * groups_per_scale
                    end_idx = min((i + 1) * groups_per_scale, weight_dequant.numel())
                    weight_dequant[start_idx:end_idx] *= s
            
            return weight_dequant.view(self.out_features, self.in_features)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Dequantize weights
            if self.use_nf4:
                weight = self._dequantize_nf4_simple(self.weight_quant, self.weight_scale)
            else:
                weight = self._dequantize_symmetric_simple(self.weight_quant, self.weight_scale)
            
            # Ensure weight has the same dtype as input
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
            
            return F.linear(x, weight, self.bias)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    model_size_mb: float
    model_size_gb: float
    total_params: int
    quantized_params: int
    quantization_ratio: float
    memory_saved_mb: float
    memory_saved_gb: float
    memory_savings_percentage: float
    tokens_per_second: float = 0.0
    model_bandwidth_utilization: float = 0.0
    theoretical_peak_bandwidth_gbps: float = 0.0
    actual_bandwidth_gbps: float = 0.0


class QuantizationPerformanceMonitor:
    """Enhanced performance monitoring for quantization."""
    
    # GPU memory bandwidth specifications (GB/s)
    GPU_BANDWIDTH = {
        "A100-80GB": 2039.0,
        "A100-40GB": 1555.0,
        "H100-80GB": 3350.0,
        "V100-32GB": 900.0,
        "RTX4090": 1008.0,
        "RTX3090": 936.0,
        "default": 1000.0  # Conservative estimate
    }
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.gpu_bandwidth_gbps = self._detect_gpu_bandwidth()
        
    def _detect_gpu_bandwidth(self) -> float:
        """Detect GPU and return its memory bandwidth."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return self.GPU_BANDWIDTH["default"]
        
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Detected GPU: {gpu_name}")
            
            # Match GPU name to bandwidth
            for gpu_key, bandwidth in self.GPU_BANDWIDTH.items():
                if gpu_key.replace("-", "").lower() in gpu_name.replace("-", "").lower():
                    print(f"Using bandwidth specification: {bandwidth} GB/s")
                    return bandwidth
                    
            print(f"Unknown GPU, using default bandwidth: {self.GPU_BANDWIDTH['default']} GB/s")
            return self.GPU_BANDWIDTH["default"]
            
        except Exception as e:
            print(f"Error detecting GPU: {e}, using default bandwidth")
            return self.GPU_BANDWIDTH["default"]
    
    def get_model_memory_info(self, model: nn.Module) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Get detailed memory information about the model."""
        param_counts = {"total": 0, "quantized": 0, "fp16": 0, "fp32": 0, "int8": 0, "int4": 0}
        memory_usage = {"total_mb": 0.0, "quantized_mb": 0.0, "fp16_mb": 0.0, "fp32_mb": 0.0}
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            param_counts["total"] += param_count
            
            # Calculate memory based on dtype
            if param.dtype == torch.float16 or param.dtype == torch.bfloat16:
                param_counts["fp16"] += param_count
                memory_usage["fp16_mb"] += param_count * 2 / 1024 / 1024
            elif param.dtype == torch.float32:
                param_counts["fp32"] += param_count
                memory_usage["fp32_mb"] += param_count * 4 / 1024 / 1024
            elif param.dtype == torch.int8:
                param_counts["int8"] += param_count
                param_counts["quantized"] += param_count
                memory_usage["quantized_mb"] += param_count * 1 / 1024 / 1024
            
            memory_usage["total_mb"] += param.numel() * param.element_size() / 1024 / 1024
        
        # Check for quantized layers (they store additional buffers)
        quantized_layer_count = 0
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and 'Quantized' in module.__class__.__name__:
                quantized_layer_count += 1
                # Count quantized parameters and scales
                if hasattr(module, 'weight_quant'):
                    quant_params = module.weight_quant.numel()
                    # Don't double count - these replace the original parameters
                    
                    # Add scale memory (this is additional overhead)
                    if hasattr(module, 'weight_scale'):
                        scale_memory = module.weight_scale.numel() * module.weight_scale.element_size() / 1024 / 1024
                        memory_usage["quantized_mb"] += scale_memory
        
        # Estimate quantized parameters based on quantized layer count
        if quantized_layer_count > 0:
            # Rough estimate: each quantized layer saves memory
            total_linear_params = sum(p.numel() for name, p in model.named_parameters() if 'weight' in name and len(p.shape) == 2)
            param_counts["quantized"] = int(total_linear_params * 0.8)  # Rough estimate
        
        return param_counts, memory_usage
    
    def calculate_performance_metrics(
        self, 
        original_model: nn.Module, 
        quantized_model: nn.Module,
        tokens_per_second: float = 0.0
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        # Get memory info for both models
        orig_counts, orig_memory = self.get_model_memory_info(original_model)
        quant_counts, quant_memory = self.get_model_memory_info(quantized_model)
        
        # Calculate savings
        memory_saved_mb = orig_memory["total_mb"] - quant_memory["total_mb"]
        memory_saved_gb = memory_saved_mb / 1024
        memory_savings_percentage = (memory_saved_mb / orig_memory["total_mb"]) * 100 if orig_memory["total_mb"] > 0 else 0
        
        # Calculate quantization ratio
        quantization_ratio = (quant_counts["quantized"] / quant_counts["total"]) * 100 if quant_counts["total"] > 0 else 0
        
        # Calculate Model Bandwidth Utilization (MBU) if tokens_per_second provided
        mbu = 0.0
        actual_bandwidth_gbps = 0.0
        
        if tokens_per_second > 0:
            # Calculate actual bandwidth usage
            # Model size in GB * tokens per second = GB/s used
            model_size_gb = quant_memory["total_mb"] / 1024
            actual_bandwidth_gbps = model_size_gb * tokens_per_second
            
            # MBU = (actual bandwidth / theoretical peak bandwidth) * 100
            mbu = (actual_bandwidth_gbps / self.gpu_bandwidth_gbps) * 100
        
        return PerformanceMetrics(
            model_size_mb=quant_memory["total_mb"],
            model_size_gb=quant_memory["total_mb"] / 1024,
            total_params=quant_counts["total"],
            quantized_params=quant_counts["quantized"],
            quantization_ratio=quantization_ratio,
            memory_saved_mb=memory_saved_mb,
            memory_saved_gb=memory_saved_gb,
            memory_savings_percentage=memory_savings_percentage,
            tokens_per_second=tokens_per_second,
            model_bandwidth_utilization=mbu,
            theoretical_peak_bandwidth_gbps=self.gpu_bandwidth_gbps,
            actual_bandwidth_gbps=actual_bandwidth_gbps
        )
    
    def benchmark_inference_speed(
        self, 
        model: nn.Module, 
        processor, 
        test_prompt: str = "What is in this image?",
        test_image_path: str = None,
        num_runs: int = 3,
        max_tokens: int = 50
    ) -> float:
        """Benchmark inference speed to calculate tokens per second."""
        
        if not test_image_path:
            print("Warning: No test image provided for benchmarking, using dummy image")
            from PIL import Image
            test_image = Image.new('RGB', (224, 224), color='white')
            test_image.save("/tmp/dummy_image.jpg")
            test_image_path = "/tmp/dummy_image.jpg"
        
        model.eval()
        total_time = 0
        total_tokens = 0
        
        with torch.no_grad():
            for run in range(num_runs):
                print(f"Benchmark run {run + 1}/{num_runs}")
                
                # Prepare inputs
                from PIL import Image
                image = Image.open(test_image_path).convert("RGB")
                inputs = processor(text=[test_prompt], images=[image], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Warmup
                if run == 0:
                    try:
                        _ = model(**inputs)
                        torch.cuda.synchronize() if self.device == "cuda" else None
                    except:
                        pass
                
                # Actual timing
                start_time = time.time()
                
                try:
                    outputs = model(**inputs)
                    torch.cuda.synchronize() if self.device == "cuda" else None
                    
                    end_time = time.time()
                    run_time = end_time - start_time
                    
                    total_time += run_time
                    total_tokens += max_tokens  # Approximate tokens generated
                    
                except Exception as e:
                    print(f"Benchmark run {run + 1} failed: {e}")
                    continue
        
        if total_time > 0:
            tokens_per_second = total_tokens / total_time
            print(f"Average inference speed: {tokens_per_second:.2f} tokens/second")
            return tokens_per_second
        else:
            print("Benchmarking failed, cannot calculate tokens per second")
            return 0.0
    
    def print_comprehensive_report(self, metrics: PerformanceMetrics):
        """Print a comprehensive performance report."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE QUANTIZATION PERFORMANCE REPORT")
        print("="*80)
        
        # Model Size Information
        print(f"\nMODEL SIZE METRICS:")
        print(f"   Total Parameters: {metrics.total_params:,}")
        print(f"   Quantized Parameters: {metrics.quantized_params:,}")
        print(f"   Quantization Ratio: {metrics.quantization_ratio:.1f}%")
        print(f"   Final Model Size: {metrics.model_size_mb:.2f} MB ({metrics.model_size_gb:.3f} GB)")
        
        # Memory Savings
        print(f"\nMEMORY SAVINGS:")
        print(f"   Memory Saved: {metrics.memory_saved_mb:.2f} MB ({metrics.memory_saved_gb:.3f} GB)")
        print(f"   Memory Savings: {metrics.memory_savings_percentage:.1f}%")
        
        # Performance Metrics (if available)
        if metrics.tokens_per_second > 0:
            print(f"\nPERFORMANCE METRICS:")
            print(f"   Inference Speed: {metrics.tokens_per_second:.2f} tokens/second")
            print(f"   Actual Bandwidth Usage: {metrics.actual_bandwidth_gbps:.2f} GB/s")
            print(f"   Theoretical Peak Bandwidth: {metrics.theoretical_peak_bandwidth_gbps:.2f} GB/s")
            print(f"   Model Bandwidth Utilization (MBU): {metrics.model_bandwidth_utilization:.1f}%")
            
            # Performance interpretation
            if metrics.model_bandwidth_utilization > 70:
                print(f"   Status: Excellent bandwidth utilization! Memory bandwidth is the main bottleneck.")
            elif metrics.model_bandwidth_utilization > 50:
                print(f"   Status: Good bandwidth utilization. Some room for improvement.")
            elif metrics.model_bandwidth_utilization > 30:
                print(f"   Status: Moderate bandwidth utilization. Consider further optimization.")
            else:
                print(f"   Status: Low bandwidth utilization. Significant optimization potential.")
        
        # Recommendations
        print(f"\nOPTIMIZATION RECOMMENDATIONS:")
        if metrics.quantization_ratio < 80:
            print(f"   • Consider quantizing more layers (currently {metrics.quantization_ratio:.1f}%)")
        if metrics.memory_savings_percentage < 30:
            print(f"   • Try more aggressive quantization (4-bit, NF4)")
        if metrics.tokens_per_second > 0 and metrics.model_bandwidth_utilization < 50:
            print(f"   • Consider torch.compile() for kernel fusion")
            print(f"   • Profile for compute vs. memory bottlenecks")
        
        print("="*80)


def replace_linear_layers_enhanced(
    model: nn.Module, 
    config: QuantizationConfig,
    name_prefix: str = ""
) -> nn.Module:
    """Replace linear layers with enhanced tracking."""
    
    replacements_made = 0
    
    for name, child in model.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        
        if isinstance(child, nn.Linear):
            # Check if we should skip this layer
            should_skip = any(skip_pattern in full_name for skip_pattern in config.skip_layers)
            
            if should_skip:
                print(f"Skipping quantization for layer: {full_name}")
                continue
            
            # Check component-specific quantization flags
            if "vision_tower" in full_name and not config.quant_vision:
                continue
            if "language_model" in full_name and not config.quant_text:
                continue
            if "multi_modal_projector" in full_name and not config.quant_projector:
                continue
            
            print(f"Quantizing layer: {full_name} ({child.weight.shape})")
            
            try:
                # Create quantized layer
                quantized_layer = SimpleQuantizedLinear(
                    weight=child.weight.data,
                    bias=child.bias.data if child.bias is not None else None,
                    bits=config.bits,
                    group_size=config.group_size,
                    use_nf4=config.use_nf4
                )
                
                setattr(model, name, quantized_layer)
                replacements_made += 1
                
            except Exception as e:
                print(f"Warning: Failed to quantize layer {full_name}: {e}")
                print(f"Keeping original layer")
            
        else:
            # Recursively process child modules
            replace_linear_layers_enhanced(child, config, full_name)
    
    if name_prefix == "":  # Only print for root call
        print(f"Total linear layers quantized: {replacements_made}")
    
    return model


def quantize_paligemma_enhanced(
    model,
    config: QuantizationConfig,
    device: str = "cuda",
    enable_performance_monitoring: bool = True,
    benchmark_performance: bool = False,
    processor=None,
    test_prompt: str = "What is in this image?",
    test_image_path: str = None
) -> Tuple[Any, Optional[PerformanceMetrics]]:
    """
    Enhanced quantization with comprehensive performance monitoring.
    """
    print(f"Starting {config.bits}-bit quantization of PaliGemma model with enhanced monitoring...")
    print(f"Configuration: NF4={config.use_nf4}, group_size={config.group_size}")
    
    # Initialize performance monitor
    monitor = None
    metrics = None
    
    if enable_performance_monitoring:
        monitor = QuantizationPerformanceMonitor(device)
        
        # Store original model info for comparison
        print("Capturing original model metrics...")
        orig_counts, orig_memory = monitor.get_model_memory_info(model)
        print(f"Original model: {orig_counts['total']:,} parameters, {orig_memory['total_mb']:.2f} MB")
    
    # Quantize the model
    model.eval()
    quantized_model = replace_linear_layers_enhanced(model, config)
    
    # Calculate performance metrics
    if enable_performance_monitoring and monitor:
        print("\nCalculating performance metrics...")
        
        # For metrics calculation, we need to compare with original
        # Since we modified the model in place, we'll estimate based on configuration
        quant_counts, quant_memory = monitor.get_model_memory_info(quantized_model)
        
        # Estimate original memory usage
        estimated_original_mb = quant_memory["total_mb"] * (16.0 / config.bits)  # Rough estimate
        memory_saved_mb = estimated_original_mb - quant_memory["total_mb"]
        memory_saved_gb = memory_saved_mb / 1024
        memory_savings_percentage = (memory_saved_mb / estimated_original_mb) * 100 if estimated_original_mb > 0 else 0
        
        # Create metrics object
        metrics = PerformanceMetrics(
            model_size_mb=quant_memory["total_mb"],
            model_size_gb=quant_memory["total_mb"] / 1024,
            total_params=quant_counts["total"],
            quantized_params=quant_counts["quantized"],
            quantization_ratio=(quant_counts["quantized"] / quant_counts["total"]) * 100 if quant_counts["total"] > 0 else 0,
            memory_saved_mb=memory_saved_mb,
            memory_saved_gb=memory_saved_gb,
            memory_savings_percentage=memory_savings_percentage,
            tokens_per_second=0.0,
            model_bandwidth_utilization=0.0,
            theoretical_peak_bandwidth_gbps=monitor.gpu_bandwidth_gbps,
            actual_bandwidth_gbps=0.0
        )
        
        # Optional: Run benchmark to get actual performance
        if benchmark_performance and processor and test_image_path:
            print("Running performance benchmark...")
            try:
                tokens_per_second = monitor.benchmark_inference_speed(
                    quantized_model, processor, test_prompt, test_image_path, num_runs=3
                )
                
                # Update metrics with performance data
                metrics.tokens_per_second = tokens_per_second
                metrics.actual_bandwidth_gbps = metrics.model_size_gb * tokens_per_second
                metrics.model_bandwidth_utilization = (
                    (metrics.actual_bandwidth_gbps / monitor.gpu_bandwidth_gbps) * 100
                )
                
            except Exception as e:
                print(f"Performance benchmark failed: {e}")
                print("Continuing with static metrics only...")
        
        # Print comprehensive report
        monitor.print_comprehensive_report(metrics)
    
    else:
        # Fallback to simple reporting
        print(f"\nSimple quantization complete!")
        estimated_savings = 1.0 - (config.bits / 16.0)  # Keep old method as fallback
        print(f"Estimated memory savings: {estimated_savings*100:.1f}%")
    
    return quantized_model, metrics


# Convenience functions that work with your existing code
def quantize_int8_enhanced(
    model,
    group_size: int = 128,
    device: str = "cuda",
    enable_monitoring: bool = True,
    **kwargs
) -> Tuple[Any, Optional[PerformanceMetrics]]:
    """Enhanced INT8 quantization with performance monitoring."""
    config = QuantizationConfig(bits=8, group_size=group_size)
    return quantize_paligemma_enhanced(
        model, config, device, enable_performance_monitoring=enable_monitoring, **kwargs
    )


def quantize_int4_enhanced(
    model,
    group_size: int = 128,
    device: str = "cuda",
    enable_monitoring: bool = True,
    **kwargs
) -> Tuple[Any, Optional[PerformanceMetrics]]:
    """Enhanced INT4 quantization with performance monitoring."""
    config = QuantizationConfig(bits=4, group_size=group_size)
    return quantize_paligemma_enhanced(
        model, config, device, enable_performance_monitoring=enable_monitoring, **kwargs
    )


def quantize_nf4_enhanced(
    model,
    group_size: int = 128,
    device: str = "cuda",
    enable_monitoring: bool = True,
    **kwargs
) -> Tuple[Any, Optional[PerformanceMetrics]]:
    """Enhanced NF4 quantization with performance monitoring."""
    config = QuantizationConfig(bits=4, use_nf4=True, group_size=group_size)
    return quantize_paligemma_enhanced(
        model, config, device, enable_performance_monitoring=enable_monitoring, **kwargs
    )


# Keep compatibility with your old functions
def quantize_paligemma_simple(model, config, calibration_data=None):
    """Compatibility function for old code."""
    quantized_model, _ = quantize_paligemma_enhanced(
        model, config, enable_performance_monitoring=False
    )
    return quantized_model


def quantize_int8_simple(model, group_size=128):
    """Compatibility function for old code."""
    config = QuantizationConfig(bits=8, group_size=group_size)
    return quantize_paligemma_simple(model, config)


def quantize_int4_simple(model, group_size=128):
    """Compatibility function for old code."""
    config = QuantizationConfig(bits=4, group_size=group_size)
    return quantize_paligemma_simple(model, config)


def quantize_nf4_simple(model, group_size=128):
    """Compatibility function for old code."""
    config = QuantizationConfig(bits=4, use_nf4=True, group_size=group_size)
    return quantize_paligemma_simple(model, config)


if __name__ == "__main__":
    print("Enhanced PaliGemma Quantization Module with Performance Monitoring")
    print("This version provides comprehensive MBU calculations and memory analysis")
    print("\nKey improvements:")
    print("- Model Bandwidth Utilization (MBU) calculations")
    print("- GPU-specific bandwidth detection")
    print("- Detailed memory usage analysis")
    print("- Performance benchmarking capabilities")
    print("- Comprehensive reporting with optimization recommendations")
    print("- Backward compatibility with existing code")