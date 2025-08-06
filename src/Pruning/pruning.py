import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
import sys

# Setup sys.path so we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_hf_model
from gemma_global import PaliGemmaForConditionalGeneration, PaliGemmaConfig  # Adjust if using SWA

# Load model and tokenizer
device = "cuda"
model_path = "/home/jupyter/Paligemma2/google/paligemma-3b-pt-896"

# Pass model + config classes
model, tokenizer = load_hf_model(
    model_path,
    device,
    paligemma_model_class=PaliGemmaForConditionalGeneration,
    paligemma_config_class=PaliGemmaConfig
)

# 2:4 Sparsity mask function
def apply_2to4_sparsity(tensor):
    tensor = tensor.clone()
    shape = tensor.shape
    if shape[-1] % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4 for 2:4 sparsity.")

    view_shape = shape[:-1] + (shape[-1] // 4, 4)
    tensor_reshaped = tensor.view(*view_shape)

    abs_tensor = tensor_reshaped.abs()
    sorted_indices = abs_tensor.argsort(dim=-1)
    mask = torch.ones_like(tensor_reshaped)
    for i in range(2):
        idx = sorted_indices[..., i].unsqueeze(-1)
        mask.scatter_(-1, idx, 0.0)

    pruned_tensor = tensor_reshaped * mask
    return pruned_tensor.view(shape)

# Traverse model and prune all Linear layers
def sparsify_model_2to4(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                print(f"Pruning {name} with 2:4 sparsity")
                pruned_weight = apply_2to4_sparsity(module.weight)
                module.weight.copy_(pruned_weight)

sparsify_model_2to4(model)

# Save pruned model
output_dir = "/home/jupyter/Paligemma2/google-pruning/paligemma-3b-2to4-pruned"
os.makedirs(output_dir, exist_ok=True)

torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
tokenizer.save_pretrained(output_dir)

print(f" Sparsified model saved to {output_dir}")
