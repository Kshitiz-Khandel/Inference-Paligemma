"""
Basic quantization module that provides the core functionality for quantized inference.
This module provides reliable quantization without the enhanced monitoring features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass
from PIL import Image


@dataclass 
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
        skip_layers: Optional[List[str]] = None
    ):
        self.bits = bits
        self.group_size = group_size
        self.calibration_samples = calibration_samples
        self.use_nf4 = use_nf4
        self.double_quant = double_quant
        self.quant_vision = quant_vision
        self.quant_text = quant_text
        self.quant_projector = quant_projector
        self.skip_layers = skip_layers or [
            "embed_tokens", "lm_head", 
            "self_attn.q_proj", "self_attn.k_proj", 
            "self_attn.v_proj", "self_attn.o_proj"  # Skip attention
        ]
        
        # Validate configuration
        if bits not in [4, 8]:
            raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")
        if use_nf4 and bits != 4:
            raise ValueError("NF4 quantization only supports 4-bit")


class QuantizedLinear(nn.Module):
    """Quantized linear layer implementation."""
    
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
        
        # Store original shape
        self.register_buffer('original_shape', torch.tensor(weight.shape))
        
        # Quantize the weight
        if use_nf4 and bits == 4:
            self.weight_quant, self.weight_scale = self._quantize_nf4(weight)
        else:
            self.weight_quant, self.weight_scale = self._quantize_symmetric(weight)
        
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
    
    def _quantize_nf4(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """NF4 quantization implementation."""
        device = weight.device
        dtype = weight.dtype
        nf4_values = self._get_nf4_values().to(device=device, dtype=dtype)
        
        weight_flat = weight.view(-1)
        
        # Group processing
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
            scale = abs_max / nf4_values.max() if abs_max > 0 else torch.tensor(1.0, device=device, dtype=dtype)
            scales.append(scale)
            
            if scale > 0:
                normalized = group / scale
                distances = torch.abs(normalized.unsqueeze(1) - nf4_values.unsqueeze(0))
                indices = torch.argmin(distances, dim=1)
                quantized_groups.append(indices)
            else:
                quantized_groups.append(torch.zeros_like(group, dtype=torch.long, device=device))
        
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
    
    def _quantize_symmetric(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric quantization implementation."""
        device = weight.device
        dtype = weight.dtype
        
        weight_flat = weight.view(-1)
        
        if self.bits == 8:
            qmax = 127
        else:  # 4-bit
            qmax = 7
        
        # Group processing
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
    
    def _dequantize_nf4(self, weight_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """NF4 dequantization."""
        device = weight_quant.device
        dtype = self.weight_scale.dtype
        nf4_values = self._get_nf4_values().to(device=device, dtype=dtype)
        
        indices = torch.clamp(weight_quant.long(), 0, 15)
        weight_dequant = nf4_values[indices]
        
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
    
    def _dequantize_symmetric(self, weight_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Symmetric dequantization."""
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
            weight = self._dequantize_nf4(self.weight_quant, self.weight_scale)
        else:
            weight = self._dequantize_symmetric(self.weight_quant, self.weight_scale)
        
        # Ensure weight has the same dtype as input
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        
        return F.linear(x, weight, self.bias)


class CalibrationDataLoader:
    """Calibration data loader for quantization."""
    
    def __init__(self, calibration_samples: int = 512):
        self.calibration_samples = calibration_samples
    
    def get_calibration_data(self, tokenizer, device: str = "cuda"):
        """Generate calibration data."""
        # Create text prompts for calibration
        prompts = [
            "What is in this image?",
            "Describe the scene in detail.",
            "What objects can you see?",
            "What colors are present?",
            "What is happening in the picture?",
        ] * (self.calibration_samples // 5 + 1)
        
        prompts = prompts[:self.calibration_samples]
        
        # Tokenize prompts
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        return tokenized


def create_calibration_data(processor, tokenizer, num_samples: int = 512, device: str = "cuda"):
    """Create calibration data for quantization."""
    # Create dummy image for calibration
    dummy_image = Image.new('RGB', (224, 224), color='white')
    
    # Create calibration prompts
    calibration_prompts = [
        "What is in this image?",
        "Describe the scene.",
        "What objects can you see?", 
        "What colors are present?",
        "What is happening here?",
    ] * (num_samples // 5 + 1)
    
    calibration_prompts = calibration_prompts[:num_samples]
    images = [dummy_image] * len(calibration_prompts)
    
    # Process with the processor
    calibration_data = processor(
        text=calibration_prompts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    calibration_data = {k: v.to(device) for k, v in calibration_data.items()}
    
    return calibration_data


def replace_linear_layers(
    model: nn.Module, 
    config: QuantizationConfig,
    name_prefix: str = ""
) -> int:
    """Replace linear layers with quantized versions."""
    
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
                quantized_layer = QuantizedLinear(
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
            replacements_made += replace_linear_layers(child, config, full_name)
    
    return replacements_made


def quantize_paligemma(
    model,
    config: QuantizationConfig,
    calibration_data=None
):
    """Basic quantization function without monitoring."""
    print(f"Starting {config.bits}-bit quantization...")
    
    model.eval()
    replacements_made = replace_linear_layers(model, config)
    print(f"Total linear layers quantized: {replacements_made}")
    
    return model


# Convenience functions
def quantize_int8(model, group_size: int = 128):
    """INT8 quantization with default settings."""
    config = QuantizationConfig(bits=8, group_size=group_size)
    return quantize_paligemma(model, config)


def quantize_int4(model, group_size: int = 128):
    """INT4 quantization with default settings."""
    config = QuantizationConfig(bits=4, group_size=group_size)
    return quantize_paligemma(model, config)


def quantize_nf4(model, group_size: int = 128):
    """NF4 quantization with default settings."""
    config = QuantizationConfig(bits=4, use_nf4=True, group_size=group_size)
    return quantize_paligemma(model, config)


if __name__ == "__main__":
    print("Quantization Module")
    print("Provides quantization functionality for PaliGemma models")