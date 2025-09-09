import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import json
from pathlib import Path
from gemma_flash import PaliGemmaForConditionalGeneration, PaliGemmaConfig, GemmaConfig
from utils import load_hf_model
import copy


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "name"):  # for AttentionType enums
        return obj.name
    else:
        return obj


class DraftModelCreator:
    """Utility class to create draft models from existing target models"""
    
    @staticmethod
    def create_layer_pruned_draft(
        target_model: PaliGemmaForConditionalGeneration,
        keep_every_n_layers: int = 2,
        keep_first_n: int = 2,
        keep_last_n: int = 2
    ) -> PaliGemmaForConditionalGeneration:
        """
        Create a draft model by keeping only a subset of transformer layers.
        
        Args:
            target_model: The full target model
            keep_every_n_layers: Keep every nth layer from the middle
            keep_first_n: Number of first layers to always keep
            keep_last_n: Number of last layers to always keep
        """
        original_config = target_model.config
        target_layers = target_model.language_model.model.layers
        total_layers = len(target_layers)
        
        # Determine which layers to keep
        layers_to_keep = []
        
        # Keep first N layers
        layers_to_keep.extend(range(min(keep_first_n, total_layers)))
        
        # Keep every nth layer from the middle
        middle_start = keep_first_n
        middle_end = max(0, total_layers - keep_last_n)
        for i in range(middle_start, middle_end, keep_every_n_layers):
            if i not in layers_to_keep:
                layers_to_keep.append(i)
        
        # Keep last N layers
        last_layers = range(max(0, total_layers - keep_last_n), total_layers)
        layers_to_keep.extend([i for i in last_layers if i not in layers_to_keep])
        
        layers_to_keep = sorted(list(set(layers_to_keep)))
        draft_num_layers = len(layers_to_keep)
        
        print(f"Creating draft model with {draft_num_layers} layers (from original {total_layers})")
        print(f"Keeping layers: {layers_to_keep}")
        
        # Create new config for draft model
        draft_text_config = copy.deepcopy(original_config.text_config.__dict__)
        draft_text_config['num_hidden_layers'] = draft_num_layers
        draft_text_config.pop('pad_token_id', None)  # remove duplicate


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
        
        # Create draft model with new config
        draft_model = PaliGemmaForConditionalGeneration(draft_config)
        
        # Copy weights
        # 1. Copy vision tower weights
        draft_model.vision_tower.load_state_dict(target_model.vision_tower.state_dict())
        
        # 2. Copy multi-modal projector weights
        draft_model.multi_modal_projector.load_state_dict(target_model.multi_modal_projector.state_dict())
        
        # 3. Copy embedding and output head weights
        draft_model.language_model.model.embed_tokens.load_state_dict(
            target_model.language_model.model.embed_tokens.state_dict()
        )
        draft_model.language_model.lm_head.load_state_dict(
            target_model.language_model.lm_head.state_dict()
        )
        draft_model.language_model.model.norm.load_state_dict(
            target_model.language_model.model.norm.state_dict()
        )
        
        # 4. Copy selected layer weights
        for draft_idx, original_idx in enumerate(layers_to_keep):
            draft_model.language_model.model.layers[draft_idx].load_state_dict(
                target_layers[original_idx].state_dict()
            )
        
        return draft_model
    
    @staticmethod
    def create_width_pruned_draft(
        target_model: PaliGemmaForConditionalGeneration,
        hidden_size_ratio: float = 0.5,
        intermediate_size_ratio: float = 0.5,
        num_heads_ratio: float = 0.5
    ) -> PaliGemmaForConditionalGeneration:
        """
        Create a draft model by reducing the width (hidden dimensions, heads) of the model.
        Note: This requires more complex weight copying and may need fine-tuning.
        """
        original_config = target_model.config
        
        # Calculate new dimensions
        new_hidden_size = int(original_config.text_config.hidden_size * hidden_size_ratio)
        new_intermediate_size = int(original_config.text_config.intermediate_size * intermediate_size_ratio)
        new_num_heads = max(1, int(original_config.text_config.num_attention_heads * num_heads_ratio))
        new_num_kv_heads = max(1, int(original_config.text_config.num_key_value_heads * num_heads_ratio))
        
        # Ensure dimensions are compatible
        new_head_dim = new_hidden_size // new_num_heads
        new_hidden_size = new_num_heads * new_head_dim  # Adjust to be divisible
        
        print(f"Creating width-pruned draft model:")
        print(f"  Hidden size: {original_config.text_config.hidden_size} -> {new_hidden_size}")
        print(f"  Intermediate size: {original_config.text_config.intermediate_size} -> {new_intermediate_size}")
        print(f"  Attention heads: {original_config.text_config.num_attention_heads} -> {new_num_heads}")
        
        # Create new config
        draft_text_config = copy.deepcopy(original_config.text_config.__dict__)
        draft_text_config.update({
            'hidden_size': new_hidden_size,
            'intermediate_size': new_intermediate_size,
            'num_attention_heads': new_num_heads,
            'num_key_value_heads': new_num_kv_heads,
            'head_dim': new_head_dim
        })
        
        draft_config = PaliGemmaConfig(
            vision_config=original_config.vision_config.__dict__,
            text_config=draft_text_config,
            ignore_index=original_config.ignore_index,
            image_token_index=original_config.image_token_index,
            vocab_size=original_config.vocab_size,
            projection_dim=new_hidden_size,  # Adjust projection to match new hidden size
            hidden_size=new_hidden_size,
            pad_token_id=original_config.pad_token_id
        )
        
        draft_model = PaliGemmaForConditionalGeneration(draft_config)
        
        print("Warning: Width-pruned models require careful weight initialization and likely need fine-tuning!")
        return draft_model
    
    @staticmethod
    def create_attention_pruned_draft(
        target_model: PaliGemmaForConditionalGeneration,
        use_only_global_attention: bool = True,
        reduce_sliding_window: bool = True,
        new_sliding_window_size: int = 1024
    ) -> PaliGemmaForConditionalGeneration:
        """
        Create a draft model by simplifying the attention mechanism.
        """
        original_config = target_model.config
        
        # Modify attention configuration
        draft_text_config = copy.deepcopy(original_config.text_config.__dict__)
        # Remove pad_token_id from text_config to avoid duplicate parameter
        draft_text_config.pop('pad_token_id', None)
        
        if use_only_global_attention:
            # Use only global attention for all layers
            from gemma_flash import AttentionType
            draft_text_config['attn_types'] = [AttentionType.GLOBAL] * draft_text_config['num_hidden_layers']
        
        if reduce_sliding_window:
            draft_text_config['sliding_window_size'] = new_sliding_window_size
        
        print(f"Creating attention-simplified draft model")
        print(f"  Global attention only: {use_only_global_attention}")
        print(f"  Sliding window: {draft_text_config.get('sliding_window_size', 'N/A')}")
        
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
        
        # Copy all weights (architecture is the same, just attention behavior changes)
        draft_model.load_state_dict(target_model.state_dict())
        
        return draft_model

    @staticmethod
    def save_draft_model(draft_model: PaliGemmaForConditionalGeneration, save_path: str):
        """Save the draft model to disk"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(draft_model.state_dict(), save_path / "pytorch_model.bin")
        
        # Save config
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
        
        with open(save_path / "config.json", 'w') as f:
            json.dump(make_json_serializable(config_dict), f, indent=2)
        
        print(f"Draft model saved to {save_path}")

    @staticmethod
    def load_draft_model(model_path: str, device: str) -> PaliGemmaForConditionalGeneration:
        """Load a draft model from disk"""
        model_path = Path(model_path)
        
        # Load config
        with open(model_path / "config.json", 'r') as f:
            config_dict = json.load(f)
        
        config = PaliGemmaConfig(**config_dict)
        draft_model = PaliGemmaForConditionalGeneration(config)
        
        # Load weights
        state_dict = torch.load(model_path / "pytorch_model.bin", map_location=device)
        draft_model.load_state_dict(state_dict)
        
        return draft_model.to(device)


def create_draft_model_from_checkpoint(
    target_model_path: str,
    draft_save_path: str,
    method: str = "layer_pruning",
    device: str = "cuda",
    **kwargs
):
    """
    Convenient function to create a draft model from a checkpoint
    
    Args:
        target_model_path: Path to target model
        draft_save_path: Where to save the draft model
        method: "layer_pruning", "width_pruning", or "attention_pruning"
        device: Device to use
        **kwargs: Additional arguments for the specific pruning method
    """
    print(f"Loading target model from {target_model_path}")
    target_model, tokenizer = load_hf_model(target_model_path, device)
    target_model = target_model.to(device).eval()
    
    creator = DraftModelCreator()
    
    if method == "layer_pruning":
        draft_model = creator.create_layer_pruned_draft(target_model, **kwargs)
    elif method == "width_pruning":
        draft_model = creator.create_width_pruned_draft(target_model, **kwargs)
    elif method == "attention_pruning":
        draft_model = creator.create_attention_pruned_draft(target_model, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Saving draft model to {draft_save_path}")
    creator.save_draft_model(draft_model, draft_save_path)
    
    # Copy tokenizer files to draft model directory
    import shutil
    target_path = Path(target_model_path)
    draft_path = Path(draft_save_path)
    
    for tokenizer_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = target_path / tokenizer_file
        if src.exists():
            shutil.copy2(src, draft_path / tokenizer_file)
    
    return draft_model


if __name__ == "__main__":
    import fire
    
    def main(
        target_model_path: str,
        draft_save_path: str,
        method: str = "layer_pruning",
        keep_every_n_layers: int = 2,
        keep_first_n: int = 2,
        keep_last_n: int = 2,
        hidden_size_ratio: float = 0.5,
        intermediate_size_ratio: float = 0.5,
        num_heads_ratio: float = 0.5,
        device: str = "cuda"
    ):
        """
        Create a draft model from target model
        
        Examples:
        # Layer pruning (recommended)
        python create_draft_model.py --target_model_path /path/to/target --draft_save_path /path/to/draft --method layer_pruning --keep_every_n_layers 2
        
        # Width pruning (needs fine-tuning)
        python create_draft_model.py --target_model_path /path/to/target --draft_save_path /path/to/draft --method width_pruning --hidden_size_ratio 0.5
        
        # Attention pruning
        python create_draft_model.py --target_model_path /path/to/target --draft_save_path /path/to/draft --method attention_pruning
        """
        
        kwargs = {}
        if method == "layer_pruning":
            kwargs.update({
                'keep_every_n_layers': keep_every_n_layers,
                'keep_first_n': keep_first_n,
                'keep_last_n': keep_last_n
            })
        elif method == "width_pruning":
            kwargs.update({
                'hidden_size_ratio': hidden_size_ratio,
                'intermediate_size_ratio': intermediate_size_ratio,
                'num_heads_ratio': num_heads_ratio
            })
        
        create_draft_model_from_checkpoint(
            target_model_path, draft_save_path, method, device, **kwargs
        )
    
    fire.Fire(main)