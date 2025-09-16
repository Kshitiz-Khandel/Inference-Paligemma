"""
 quantization module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import time
from dataclasses import dataclass
from PIL import Image
import math


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


class OptimizedQuantizationConfig:
    """Optimized configuration for quantization settings."""
    
    def __init__(
        self,
        bits: int = 8,
        group_size: int = 64,
        calibration_samples: int = 256,
        use_nf4: bool = False,
        double_quant: bool = False,
        quant_vision: bool = True,
        quant_text: bool = True,
        quant_projector: bool = True,
        skip_layers: Optional[List[str]] = None,
        use_fast_kernels: bool = True,
        optimize_for_inference: bool = True,
        fuse_operations: bool = True
    ):
        self.bits = bits
        self.group_size = group_size
        self.calibration_samples = calibration_samples
        self.use_nf4 = use_nf4
        self.double_quant = double_quant
        self.quant_vision = quant_vision
        self.quant_text = quant_text
        self.quant_projector = quant_projector
        self.skip_layers = skip_layers or ["embed_tokens", "lm_head"]
        self.use_fast_kernels = use_fast_kernels
        self.optimize_for_inference = optimize_for_inference
        self.fuse_operations = fuse_operations
        
        if bits not in [4, 8]:
            raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")
        if use_nf4 and bits != 4:
            raise ValueError("NF4 quantization only supports 4-bit")


class OptimizedQuantizedLinear(nn.Module):
    """Fixed quantized linear layer with proper shape handling."""
    
    def __init__(
        self, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor] = None,
        bits: int = 8,
        group_size: int = 64,
        use_nf4: bool = False,
        use_fast_kernels: bool = True
    ):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.use_nf4 = use_nf4
        self.use_fast_kernels = use_fast_kernels
        self.out_features, self.in_features = weight.shape
        
        # Quantize and store
        if use_nf4 and bits == 4:
            self.weight_quant, self.weight_scale, self.weight_zero_point = self._quantize_nf4_fixed(weight)
        else:
            self.weight_quant, self.weight_scale, self.weight_zero_point = self._quantize_symmetric_fixed(weight)
        
        # Store bias
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
    
    def _quantize_symmetric_fixed(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed symmetric quantization with proper shape handling."""
        device = weight.device
        dtype = weight.dtype
        original_shape = weight.shape
        
        # Flatten weight for group-wise quantization
        weight_flat = weight.view(-1)
        
        if weight_flat.numel() < self.group_size:
            # If total elements less than group size, use entire weight as one group
            groups = [weight_flat]
        else:
            # Split into groups
            num_groups = (weight_flat.numel() + self.group_size - 1) // self.group_size
            # Pad if necessary
            if weight_flat.numel() % self.group_size != 0:
                pad_size = num_groups * self.group_size - weight_flat.numel()
                weight_flat = F.pad(weight_flat, (0, pad_size))
            
            groups = weight_flat.view(num_groups, self.group_size)
        
        if self.bits == 8:
            qmax = 127
        else:  # 4-bit
            qmax = 7
        
        if len(groups) == 1 and groups[0].numel() < self.group_size:
            # Single group case
            group = groups[0]
            abs_max = torch.max(torch.abs(group))
            scale = abs_max / qmax if abs_max > 0 else torch.tensor(1.0, device=device, dtype=dtype)
            quantized = torch.round(group / scale).clamp(-qmax, qmax)
            
            # Restore original shape
            if quantized.numel() < weight.numel():
                quantized = F.pad(quantized, (0, weight.numel() - quantized.numel()))
            quantized = quantized[:weight.numel()].view(original_shape).to(torch.int8)
            
            # Scale should be scalar for broadcasting
            scale = scale.unsqueeze(0)  # Make it [1] instead of scalar
            
        else:
            # Multiple groups case
            scales = []
            quantized_groups = []
            
            for group in groups:
                abs_max = torch.max(torch.abs(group))
                scale = abs_max / qmax if abs_max > 0 else torch.tensor(1.0, device=device, dtype=dtype)
                quantized_group = torch.round(group / scale).clamp(-qmax, qmax)
                
                scales.append(scale)
                quantized_groups.append(quantized_group)
            
            # Combine results
            quantized = torch.cat(quantized_groups)[:weight.numel()].view(original_shape).to(torch.int8)
            scale = torch.stack(scales)
        
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
        return quantized, scale, zero_point
    
    def _quantize_nf4_fixed(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed NF4 quantization."""
        device = weight.device
        dtype = weight.dtype
        
        nf4_values = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ], device=device, dtype=dtype)
        
        weight_flat = weight.view(-1)
        num_groups = (weight_flat.numel() + self.group_size - 1) // self.group_size
        
        # Pad if necessary
        if weight_flat.numel() % self.group_size != 0:
            pad_size = num_groups * self.group_size - weight_flat.numel()
            weight_flat = F.pad(weight_flat, (0, pad_size))
        
        weight_grouped = weight_flat.view(num_groups, self.group_size)
        
        # NF4 quantization
        scales = []
        indices_list = []
        
        for group in weight_grouped:
            abs_max = torch.max(torch.abs(group))
            scale = abs_max / nf4_values.max() if abs_max > 0 else torch.tensor(1.0, device=device, dtype=dtype)
            normalized = group / scale
            
            # Find closest NF4 values
            distances = torch.abs(normalized.unsqueeze(-1) - nf4_values.unsqueeze(0))
            indices = torch.argmin(distances, dim=-1)
            
            scales.append(scale)
            indices_list.append(indices)
        
        # Combine results
        all_indices = torch.cat(indices_list)[:weight.numel()]
        indices = all_indices.view(weight.shape).to(torch.int8)
        scale = torch.stack(scales)
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
        
        return indices, scale, zero_point
    
    def _dequantize_symmetric_fixed(self, weight_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Fixed symmetric dequantization with proper broadcasting."""
        weight_float = weight_quant.float()
        
        if scale.numel() == 1:
            # Single scale - broadcast to entire weight
            return weight_float * scale.item()
        elif scale.numel() == weight_quant.shape[0]:
            # Per-output-channel scaling
            return weight_float * scale.view(-1, 1)
        else:
            # Group-wise scaling - need to reshape and broadcast properly
            out_features, in_features = weight_quant.shape
            weight_flat = weight_float.view(-1)
            
            # Calculate elements per group
            elements_per_group = self.group_size
            num_groups = scale.numel()
            
            # Reconstruct grouped scaling
            result_flat = torch.zeros_like(weight_flat)
            for i, group_scale in enumerate(scale):
                start_idx = i * elements_per_group
                end_idx = min(start_idx + elements_per_group, weight_flat.numel())
                result_flat[start_idx:end_idx] = weight_flat[start_idx:end_idx] * group_scale
            
            return result_flat.view(out_features, in_features)
    
    def _dequantize_nf4_fixed(self, weight_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Fixed NF4 dequantization."""
        nf4_values = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ], device=weight_quant.device, dtype=scale.dtype)
        
        indices = torch.clamp(weight_quant.long(), 0, 15)
        weight = nf4_values[indices]
        
        # Apply group-wise scaling similar to symmetric case
        if scale.numel() == 1:
            return weight * scale.item()
        else:
            weight_flat = weight.view(-1)
            result_flat = torch.zeros_like(weight_flat)
            elements_per_group = self.group_size
            
            for i, group_scale in enumerate(scale):
                start_idx = i * elements_per_group
                end_idx = min(start_idx + elements_per_group, weight_flat.numel())
                result_flat[start_idx:end_idx] = weight_flat[start_idx:end_idx] * group_scale
            
            return result_flat.view(weight.shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fixed dequantization."""
        # Dequantize weights
        if self.use_nf4:
            weight = self._dequantize_nf4_fixed(self.weight_quant, self.weight_scale)
        else:
            weight = self._dequantize_symmetric_fixed(self.weight_quant, self.weight_scale)
        
        # Ensure correct dtype
        weight = weight.to(x.dtype)
        
        # Standard linear operation
        return F.linear(x, weight, self.bias)


class OptimizedPerformanceMonitor:
    """Performance monitoring with corrected metrics calculation."""
    
    GPU_BANDWIDTH = {
        "H100": 3350.0,
        "H10080GB": 3350.0,
        "H100-80GB": 3350.0,
        "A100-80GB": 2039.0,
        "A100-40GB": 1555.0,
        "V100-32GB": 900.0,
        "RTX4090": 1008.0,
        "RTX3090": 936.0,
        "default": 1000.0
    }
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.gpu_bandwidth_gbps = self._detect_gpu_bandwidth()
        
    def _detect_gpu_bandwidth(self) -> float:
        """GPU detection with proper bandwidth recognition."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return self.GPU_BANDWIDTH["default"]
        
        try:
            gpu_name = torch.cuda.get_device_name(0).upper()
            print(f"Detected GPU: {gpu_name}")
            
            for gpu_key, bandwidth in self.GPU_BANDWIDTH.items():
                if gpu_key.upper().replace("-", "").replace(" ", "") in gpu_name.replace("-", "").replace(" ", ""):
                    print(f"Using bandwidth specification: {bandwidth} GB/s")
                    return bandwidth
            
            if "H100" in gpu_name:
                return self.GPU_BANDWIDTH["H100"]
                    
            print(f"Unknown GPU, using default bandwidth: {self.GPU_BANDWIDTH['default']} GB/s")
            return self.GPU_BANDWIDTH["default"]
            
        except Exception as e:
            print(f"Error detecting GPU: {e}")
            return self.GPU_BANDWIDTH["default"]
    
    def get_corrected_model_memory_info(self, model: nn.Module) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Corrected memory calculation."""
        param_counts = {"total": 0, "quantized": 0, "original_linear": 0}
        memory_usage = {"total_mb": 0.0, "quantized_mb": 0.0, "original_mb": 0.0}
        
        total_quantized_params = 0
        total_original_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, OptimizedQuantizedLinear):
                # Count the original parameters that this quantized layer represents
                original_params = module.out_features * module.in_features
                if module.bias is not None:
                    original_params += module.out_features
                
                total_original_params += original_params
                
                # Count actual quantized storage
                quantized_storage = module.weight_quant.numel()
                scale_storage = module.weight_scale.numel() if hasattr(module, 'weight_scale') else 0
                
                total_quantized_params += quantized_storage
                
                # Calculate memory usage
                quant_memory = (quantized_storage * module.weight_quant.element_size() + 
                              scale_storage * (module.weight_scale.element_size() if hasattr(module, 'weight_scale') else 0))
                memory_usage["quantized_mb"] += quant_memory / (1024 * 1024)
                
            elif isinstance(module, nn.Linear):
                # Regular linear layer
                params = module.weight.numel()
                if module.bias is not None:
                    params += module.bias.numel()
                total_original_params += params
                
                # Calculate memory
                memory = (module.weight.numel() * module.weight.element_size() +
                         (module.bias.numel() * module.bias.element_size() if module.bias is not None else 0))
                memory_usage["original_mb"] += memory / (1024 * 1024)
        
        param_counts["total"] = total_original_params
        param_counts["quantized"] = total_quantized_params
        param_counts["original_linear"] = total_original_params - total_quantized_params
        
        memory_usage["total_mb"] = memory_usage["quantized_mb"] + memory_usage["original_mb"]
        
        return param_counts, memory_usage
    
    def calculate_corrected_metrics(
        self, 
        original_model_size_mb: float,
        quantized_model: nn.Module,
        tokens_per_second: float = 0.0
    ) -> PerformanceMetrics:
        """Corrected performance metrics calculation."""
        
        counts, memory = self.get_corrected_model_memory_info(quantized_model)
        
        actual_model_size_mb = memory["total_mb"]
        memory_saved_mb = original_model_size_mb - actual_model_size_mb
        memory_saved_gb = memory_saved_mb / 1024
        memory_savings_percentage = (memory_saved_mb / original_model_size_mb) * 100 if original_model_size_mb > 0 else 0
        
        quantization_ratio = (counts["quantized"] / counts["total"]) * 100 if counts["total"] > 0 else 0
        
        mbu = 0.0
        actual_bandwidth_gbps = 0.0
        
        if tokens_per_second > 0:
            model_size_gb = actual_model_size_mb / 1024
            actual_bandwidth_gbps = model_size_gb * tokens_per_second
            mbu = (actual_bandwidth_gbps / self.gpu_bandwidth_gbps) * 100
        
        return PerformanceMetrics(
            model_size_mb=actual_model_size_mb,
            model_size_gb=actual_model_size_mb / 1024,
            total_params=counts["total"],
            quantized_params=counts["quantized"],
            quantization_ratio=quantization_ratio,
            memory_saved_mb=memory_saved_mb,
            memory_saved_gb=memory_saved_gb,
            memory_savings_percentage=memory_savings_percentage,
            tokens_per_second=tokens_per_second,
            model_bandwidth_utilization=mbu,
            theoretical_peak_bandwidth_gbps=self.gpu_bandwidth_gbps,
            actual_bandwidth_gbps=actual_bandwidth_gbps
        )
    
    def print_optimized_report(self, metrics: PerformanceMetrics):
        """Print optimized performance report."""
        
        print("\n" + "="*80)
        print("OPTIMIZED QUANTIZATION PERFORMANCE REPORT")
        print("="*80)
        
        print(f"\nMODEL SIZE METRICS:")
        print(f"   Total Parameters: {metrics.total_params:,}")
        print(f"   Quantized Storage: {metrics.quantized_params:,}")
        print(f"   Quantization Coverage: {metrics.quantization_ratio:.1f}% of parameters")
        print(f"   Final Model Size: {metrics.model_size_mb:.2f} MB ({metrics.model_size_gb:.3f} GB)")
        
        print(f"\nMEMORY SAVINGS:")
        print(f"   Memory Saved: {metrics.memory_saved_mb:.2f} MB ({metrics.memory_saved_gb:.3f} GB)")
        print(f"   Memory Savings: {metrics.memory_savings_percentage:.1f}%")
        
        if metrics.tokens_per_second > 0:
            print(f"\nPERFORMANCE ANALYSIS:")
            print(f"   Inference Speed: {metrics.tokens_per_second:.2f} tokens/second")
            print(f"   Actual Bandwidth Usage: {metrics.actual_bandwidth_gbps:.2f} GB/s")
            print(f"   Theoretical Peak Bandwidth: {metrics.theoretical_peak_bandwidth_gbps:.2f} GB/s")
            print(f"   Model Bandwidth Utilization (MBU): {metrics.model_bandwidth_utilization:.2f}%")
        
        print("="*80)


def replace_with_optimized_quantized_layers(
    model: nn.Module, 
    config: OptimizedQuantizationConfig,
    name_prefix: str = ""
) -> int:
    """Replace linear layers with fixed quantized versions."""
    
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
                # Create optimized quantized layer
                quantized_layer = OptimizedQuantizedLinear(
                    weight=child.weight.data,
                    bias=child.bias.data if child.bias is not None else None,
                    bits=config.bits,
                    group_size=config.group_size,
                    use_nf4=config.use_nf4,
                    use_fast_kernels=config.use_fast_kernels
                )
                
                setattr(model, name, quantized_layer)
                replacements_made += 1
                
            except Exception as e:
                print(f"Warning: Failed to quantize layer {full_name}: {e}")
        else:
            # Recursively process child modules
            replacements_made += replace_with_optimized_quantized_layers(child, config, full_name)
    
    return replacements_made


def quantize_with_optimizations(
    model,
    config: OptimizedQuantizationConfig,
    device: str = "cuda",
    enable_performance_monitoring: bool = True,
    benchmark_performance: bool = False,
    processor=None,
    test_prompt: str = "What is in this image?",
    test_image_path: str = None
) -> Tuple[Any, Optional[PerformanceMetrics]]:
    """Fixed quantization with proper shape handling."""
    
    print(f"Starting optimized {config.bits}-bit quantization with shape fixes...")
    
    # Initialize performance monitor
    monitor = None
    metrics = None
    original_model_size_mb = 0.0
    
    if enable_performance_monitoring:
        monitor = OptimizedPerformanceMonitor(device)
        original_model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        print(f"Original model size: {original_model_size_mb:.2f} MB")
    
    # Apply quantization
    model.eval()
    replacements_made = replace_with_optimized_quantized_layers(model, config)
    print(f"Total linear layers quantized: {replacements_made}")
    
    # Calculate metrics after optimization
    if enable_performance_monitoring and monitor:
        print("\nCalculating performance metrics...")
        metrics = monitor.calculate_corrected_metrics(original_model_size_mb, model, tokens_per_second=0.0)
        monitor.print_optimized_report(metrics)
    
    return model, metrics


# Convenience functions
def quantize_int8_optimized(model, group_size: int = 64, device: str = "cuda", **kwargs):
    """Fixed INT8 quantization."""
    config = OptimizedQuantizationConfig(bits=8, group_size=group_size, use_fast_kernels=False)
    return quantize_with_optimizations(model, config, device, **kwargs)


def quantize_int4_optimized(model, group_size: int = 64, device: str = "cuda", **kwargs):
    """Fixed INT4 quantization."""
    config = OptimizedQuantizationConfig(bits=4, group_size=group_size, use_fast_kernels=False)
    return quantize_with_optimizations(model, config, device, **kwargs)


def quantize_nf4_optimized(model, group_size: int = 64, device: str = "cuda", **kwargs):
    """Fixed NF4 quantization."""
    config = OptimizedQuantizationConfig(bits=4, use_nf4=True, group_size=group_size, use_fast_kernels=False)
    return quantize_with_optimizations(model, config, device, **kwargs)