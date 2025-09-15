import torch
import torch.nn as nn
import json
import copy
import shutil
from pathlib import Path
import fire
import numpy as np
from typing import Optional, List, Tuple
import logging

from gemma_flash import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from utils import load_hf_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_model_layers(model: PaliGemmaForConditionalGeneration, model_name: str):
    """Diagnose model layer weights and activations."""
    logger.info(f"\n=== DIAGNOSING {model_name.upper()} ===")
    
    layers = model.language_model.model.layers
    logger.info(f"Number of layers: {len(layers)}")
    
    # Check layer norms
    layer_norm_scales = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'input_layernorm'):
            norm_weight = layer.input_layernorm.weight
            layer_norm_scales.append(norm_weight.abs().mean().item())
    
    logger.info(f"Layer norm scales - Mean: {np.mean(layer_norm_scales):.4f}, Std: {np.std(layer_norm_scales):.4f}")
    logger.info(f"Layer norm scales - Min: {np.min(layer_norm_scales):.4f}, Max: {np.max(layer_norm_scales):.4f}")
    
    # Check attention weights
    attn_scales = []
    for i, layer in enumerate(layers):
        if hasattr(layer.self_attn, 'q_proj'):
            q_weight = layer.self_attn.q_proj.weight
            attn_scales.append(q_weight.abs().mean().item())
    
    logger.info(f"Attention scales - Mean: {np.mean(attn_scales):.4f}, Std: {np.std(attn_scales):.4f}")
    
    # Check MLP weights
    mlp_scales = []
    for i, layer in enumerate(layers):
        if hasattr(layer.mlp, 'up_proj'):
            up_weight = layer.mlp.up_proj.weight
            mlp_scales.append(up_weight.abs().mean().item())
    
    logger.info(f"MLP scales - Mean: {np.mean(mlp_scales):.4f}, Std: {np.std(mlp_scales):.4f}")
    
    # Check final layer norm
    if hasattr(model.language_model.model, 'norm'):
        final_norm = model.language_model.model.norm.weight
        logger.info(f"Final norm scale: {final_norm.abs().mean().item():.4f}")
    
    return {
        'layer_norm_scales': layer_norm_scales,
        'attn_scales': attn_scales,
        'mlp_scales': mlp_scales
    }


def test_model_forward_pass(model: PaliGemmaForConditionalGeneration, model_name: str, device: str):
    """Test forward pass and check for issues."""
    logger.info(f"\n=== TESTING {model_name.upper()} FORWARD PASS ===")
    
    model.eval()
    test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    attention_mask = torch.ones_like(test_input, device=device)
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=test_input, attention_mask=attention_mask)
            logits = outputs['logits'][0, -1]  # Last token logits
            
            logger.info(f"Logits shape: {logits.shape}")
            logger.info(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            logger.info(f"Logits mean: {logits.mean().item():.3f}, std: {logits.std().item():.3f}")
            
            # Check for NaN or Inf
            if torch.isnan(logits).any():
                logger.error(" NaN values detected in logits!")
            if torch.isinf(logits).any():
                logger.error(" Inf values detected in logits!")
            
            # Test probabilities
            probs = torch.softmax(logits, dim=-1)
            max_prob = probs.max().item()
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            
            logger.info(f"Max probability: {max_prob:.6f}")
            logger.info(f"Entropy: {entropy:.6f}")
            
            if max_prob > 0.99:
                logger.error(f" Degenerate probability distribution detected!")
                # Find the problematic token
                max_idx = probs.argmax().item()
                logger.info(f"Dominant token ID: {max_idx}")
            elif max_prob > 0.8:
                logger.warning(f" High max probability: {max_prob:.6f}")
            
            if entropy < 1.0:
                logger.warning(f" Low entropy: {entropy:.6f}")
            
            return {
                'logits_range': (logits.min().item(), logits.max().item()),
                'logits_stats': (logits.mean().item(), logits.std().item()),
                'max_prob': max_prob,
                'entropy': entropy,
                'has_nan': torch.isnan(logits).any().item(),
                'has_inf': torch.isinf(logits).any().item()
            }
            
        except Exception as e:
            logger.error(f" Forward pass failed: {e}")
            return None


def apply_gentle_fixes(draft_model: PaliGemmaForConditionalGeneration, 
                      target_model: PaliGemmaForConditionalGeneration):
    """Apply gentler fixes to the draft model."""
    logger.info("🔧 Applying gentle draft model fixes...")
    
    # Get target model statistics for reference
    target_layers = target_model.language_model.model.layers
    target_final_norm = target_model.language_model.model.norm.weight
    
    # Calculate target layer norm statistics from corresponding layers
    target_layer_norms = []
    draft_layers = draft_model.language_model.model.layers
    
    for i in range(len(draft_layers)):
        if i < len(target_layers):
            target_norm = target_layers[i].input_layernorm.weight
            target_layer_norms.append(target_norm.abs().mean().item())
    
    avg_target_norm = np.mean(target_layer_norms)
    
    # Fix 1: Use target model's layer norm values instead of resetting to 1.0
    for i, layer in enumerate(draft_layers):
        if i < len(target_layers):
            # Copy the exact layer norm from target
            layer.input_layernorm.weight.data = target_layers[i].input_layernorm.weight.data.clone()
            if hasattr(layer, 'post_attention_layernorm'):
                layer.post_attention_layernorm.weight.data = target_layers[i].post_attention_layernorm.weight.data.clone()
    
    # Fix 2: Use target model's final layer norm (scaled appropriately)
    # Instead of setting to 1.0, use the target's final norm but scaled
    draft_model.language_model.model.norm.weight.data = target_final_norm.data.clone()
    
    # Fix 3: Don't scale weights - they're already copied correctly
    # The issue might be the layer norm changes, not the weights
    
    logger.info(f" Applied gentle fixes - kept original layer norms from target model")


def create_early_exit_draft(
    target_model: PaliGemmaForConditionalGeneration,
    exit_layer: int = 9,
    apply_fixes: bool = True
) -> PaliGemmaForConditionalGeneration:
    """Create a better early exit draft model with minimal modifications."""
    original_config = target_model.config
    target_layers = target_model.language_model.model.layers
    
    logger.info(f"Creating better early exit draft at layer {exit_layer}")
    
    # Create new config
    draft_text_config = copy.deepcopy(original_config.text_config.__dict__)
    draft_text_config['num_hidden_layers'] = exit_layer
    draft_text_config.pop('pad_token_id', None)

    draft_config = PaliGemmaConfig(
        vision_config=original_config.vision_config.__dict__,
        text_config=draft_text_config,
        ignore_index=original_config.ignore_index,
        image_token_index=original_config.image_token_index,
        vocab_size=original_config.vocab_size,
        projection_dim=original_config.projection_dim,
        hidden_size=original_config.hidden_size,
        pad_token_id=original_config.pad_token_id
    )

    draft_model = PaliGemmaForConditionalGeneration(draft_config)
    
    # Copy all components exactly
    draft_model.vision_tower.load_state_dict(target_model.vision_tower.state_dict(), strict=True)
    draft_model.multi_modal_projector.load_state_dict(target_model.multi_modal_projector.state_dict(), strict=True)
    draft_model.language_model.model.embed_tokens.load_state_dict(
        target_model.language_model.model.embed_tokens.state_dict(), strict=True
    )
    draft_model.language_model.lm_head.load_state_dict(
        target_model.language_model.lm_head.state_dict(), strict=True
    )
    draft_model.language_model.model.norm.load_state_dict(
        target_model.language_model.model.norm.state_dict(), strict=True
    )
    
    # Copy first N layers exactly
    for i in range(exit_layer):
        logger.info(f"Copying layer {i}")
        draft_model.language_model.model.layers[i].load_state_dict(
            target_layers[i].state_dict(), strict=True
        )
    
    # Apply fixes only if requested
    if apply_fixes:
        apply_gentle_fixes(draft_model, target_model)
    
    return draft_model


def create_no_fix_draft(
    target_model: PaliGemmaForConditionalGeneration,
    exit_layer: int = 9
) -> PaliGemmaForConditionalGeneration:
    """Create a draft model with NO fixes applied - pure early exit."""
    logger.info(f"Creating NO-FIX early exit draft at layer {exit_layer}")
    return create_early_exit_draft(target_model, exit_layer, apply_fixes=False)


def make_json_serializable(obj):
    """Make object JSON serializable."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "name"):  # for enums
        return obj.name
    else:
        return obj


def save_draft_model(draft_model, draft_save_path: str, target_model_path: str):
    """Save the draft model to disk."""
    draft_save_path = Path(draft_save_path)
    draft_save_path.mkdir(parents=True, exist_ok=True)

    torch.save(draft_model.state_dict(), draft_save_path / "pytorch_model.bin")

    config_dict = {
        'vision_config': draft_model.config.vision_config.__dict__,
        'text_config': draft_model.config.text_config.__dict__,
        'ignore_index': draft_model.config.ignore_index,
        'image_token_index': draft_model.config.image_token_index,
        'vocab_size': draft_model.config.vocab_size,
        'projection_dim': draft_model.config.projection_dim,
        'hidden_size': draft_model.config.hidden_size,
        'pad_token_id': draft_model.config.pad_token_id
    }
    with open(draft_save_path / "config.json", "w") as f:
        json.dump(make_json_serializable(config_dict), f, indent=2)

    # Copy tokenizer files
    for f_name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = Path(target_model_path) / f_name
        if src.exists():
            shutil.copy2(src, draft_save_path / f_name)

    logger.info(f"Draft model saved to {draft_save_path}")


def main(
    target_model_path: str,
    draft_save_path: str,
    target_num_layers: int = 9,
    diagnose_only: bool = False,
    apply_fixes: bool = True,
    no_fixes: bool = False,
    device: str = "cuda"
):
    """
    Create and diagnose draft models with improved approach.
    
    Args:
        target_model_path: Path to the target model
        draft_save_path: Where to save the draft model  
        target_num_layers: Number of layers for draft model
        diagnose_only: Only run diagnostics, don't create model
        apply_fixes: Apply gentle initialization fixes
        no_fixes: Create model with no fixes (pure early exit)
        device: Device to use
    """
    logger.info(f"Loading target model from {target_model_path}")
    target_model, _ = load_hf_model(target_model_path, device)
    target_model = target_model.to(device).eval()
    
    # Diagnose target model
    target_stats = diagnose_model_layers(target_model, "target")
    target_forward_stats = test_model_forward_pass(target_model, "target", device)
    
    if diagnose_only:
        logger.info("Diagnosis complete. Exiting.")
        return
    
    # Create draft model
    if no_fixes:
        draft_model = create_no_fix_draft(target_model, target_num_layers)
        logger.info("Created draft model with NO fixes applied")
    else:
        draft_model = create_early_exit_draft(
            target_model, 
            target_num_layers, 
            apply_fixes=apply_fixes
        )
    
    draft_model = draft_model.to(device).eval()
    
    # Diagnose draft model
    draft_stats = diagnose_model_layers(draft_model, "draft")
    draft_forward_stats = test_model_forward_pass(draft_model, "draft", device)
    
    if draft_forward_stats is None:
        logger.error(" Draft model forward pass failed!")
        return
    
    # Compare models
    logger.info("\n=== MODEL COMPARISON ===")
    logger.info(f"Target logits range: {target_forward_stats['logits_range']}")
    logger.info(f"Draft logits range: {draft_forward_stats['logits_range']}")
    logger.info(f"Target max prob: {target_forward_stats['max_prob']:.6f}")
    logger.info(f"Draft max prob: {draft_forward_stats['max_prob']:.6f}")
    logger.info(f"Target entropy: {target_forward_stats['entropy']:.6f}")
    logger.info(f"Draft entropy: {draft_forward_stats['entropy']:.6f}")
    
    # More lenient validation criteria
    validation_passed = True
    issues = []
    
    if draft_forward_stats['max_prob'] > 0.95:
        validation_passed = False
        issues.append(f"Max probability too high: {draft_forward_stats['max_prob']:.6f}")
    
    if draft_forward_stats['entropy'] < 0.5:
        validation_passed = False
        issues.append(f"Entropy too low: {draft_forward_stats['entropy']:.6f}")
    
    if draft_forward_stats['has_nan'] or draft_forward_stats['has_inf']:
        validation_passed = False
        issues.append("NaN or Inf values detected")
    
    # Check if logits are extremely different
    target_logits_range = target_forward_stats['logits_range'][1] - target_forward_stats['logits_range'][0]
    draft_logits_range = draft_forward_stats['logits_range'][1] - draft_forward_stats['logits_range'][0]
    
    if draft_logits_range > target_logits_range * 5:
        issues.append(f"Logits range too large: {draft_logits_range:.2f} vs target {target_logits_range:.2f}")
    
    if validation_passed:
        logger.info(" Draft model passed validation!")
        save_draft_model(draft_model, draft_save_path, target_model_path)
    else:
        logger.warning(" Draft model has some issues but might still be usable:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        
        # Ask user if they want to save anyway
        response = input("Save model anyway? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            save_draft_model(draft_model, draft_save_path, target_model_path)
            logger.info(" Draft model saved despite validation issues")
        else:
            logger.info(" Draft model not saved")
            
        # Suggest alternatives
        if no_fixes:
            logger.info("Try running with --apply_fixes=True for better results")
        elif apply_fixes:
            logger.info("Try running with --no_fixes=True for a pure early exit approach")
        
        logger.info(f" Consider trying fewer layers (current: {target_num_layers})")


if __name__ == "__main__":
    fire.Fire(main)