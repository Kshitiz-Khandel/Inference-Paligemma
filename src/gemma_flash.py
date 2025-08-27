import torch
from torch import nn
from typing import Optional, Tuple, List
import math
import enum
from siglip import SiglipVisionConfig, SiglipVisionModel
import torch.nn.functional as F

# 1. Import the specific FlashAttention function for variable length sequences
try:
    from flash_attn import flash_attn_varlen_func
    _flash_attn_available = True
    print("✅ FlashAttention is available and will be used.")
except ImportError:
    print("⚠️ FlashAttention is not installed. Falling back to default PyTorch attention.")
    _flash_attn_available = False


class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if not self.key_cache:
            return 0
        return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        sliding_window_size=4096,
        attn_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.sliding_window_size = sliding_window_size
        if attn_types is None:
            assert self.num_hidden_layers % 2 == 0
            self.attn_types = [AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * int(self.num_hidden_layers / 2)
        else:
            self.attn_types = attn_types


class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        if position_ids.ndim == 2:
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        else:
            inv_freq_expanded = self.inv_freq[None, :].float()
            position_ids_expanded = position_ids[:, None].float()
            freqs = (position_ids_expanded @ inv_freq_expanded)

        device_type = x.device.type
        device_type = "cpu" if isinstance(device_type, str) and device_type == "mps" else device_type
        with torch.autocast(device_type=device_type, enabled=False):
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None, attn_type: AttentionType = AttentionType.GLOBAL):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_type = attn_type
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window_size = config.sliding_window_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings)

    def _flash_attn_forward(self, hidden_states, position_ids, cu_seqlens, max_seqlen):
        # Ensure position_ids is int32 for flash attention
        if position_ids.dtype != torch.int32:
            position_ids = position_ids.to(torch.int32)
            
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # Set window size for sliding window attention
        window_size = (0, 0)
        if self.attn_type == AttentionType.LOCAL_SLIDING:
            window_size = (self.sliding_window_size - 1, 0)

        output = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=True, window_size=window_size
        )
        print("Flash atteb ouput")
        return self.o_proj(output.view(-1, self.num_heads * self.head_dim))

    def _create_sliding_window_mask(self, padding_mask, seq_len):
        """Create sliding window mask properly sized"""
        device = padding_mask.device
        batch_size = padding_mask.shape[0]
        
        # Get dtype from first available tensor
        dtype = torch.float16 if padding_mask.dtype == torch.bool else padding_mask.dtype
        
        # Create causal mask first
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        
        # Create sliding window mask
        positions = torch.arange(seq_len, device=device)
        sliding_mask = (positions[:, None] - positions[None, :]) > self.sliding_window_size
        
        # Combine masks: True where attention should be blocked
        combined_mask = causal_mask | sliding_mask
        
        # Convert to attention mask format (batch_size, 1, seq_len, seq_len)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)
        mask = mask.masked_fill(combined_mask[None, None, :, :], torch.finfo(dtype).min)
        
        # Apply padding mask - ensure proper dimensions
        if padding_mask is not None:
            if padding_mask.dtype == torch.bool:
                padding_invalid = padding_mask == False
            else:
                padding_invalid = (padding_mask == 0)
                
            # Mask out attention FROM padding positions (rows)
            mask = mask.masked_fill(padding_invalid[:, None, :, None], torch.finfo(dtype).min)
            # Mask out attention TO padding positions (columns)  
            mask = mask.masked_fill(padding_invalid[:, None, None, :], torch.finfo(dtype).min)
        
        return mask

    def _eager_forward(self, hidden_states, attention_mask, position_ids, kv_cache):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Handle sliding window attention for SDPA
        final_attention_mask = attention_mask
        if self.attn_type == AttentionType.LOCAL_SLIDING and self.sliding_window_size is not None:
            if kv_cache is not None and kv_cache.num_items() > 0:
                # During decoding, use the original attention mask
                final_attention_mask = attention_mask
            else:
                # During prefill, create sliding window mask
                # Extract 1D padding mask from 4D attention mask if needed
                if attention_mask is not None and attention_mask.ndim == 4:
                    padding_mask_1d = attention_mask[:, 0, 0, :]
                elif attention_mask is not None and attention_mask.ndim == 2:
                    padding_mask_1d = attention_mask
                else:
                    # Create dummy padding mask if none provided
                    padding_mask_1d = torch.ones(bsz, q_len, device=hidden_states.device)
                    
                final_attention_mask = self._create_sliding_window_mask(padding_mask_1d, q_len)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=final_attention_mask, 
            is_causal=(final_attention_mask is None)
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None, **kwargs):
        if "cu_seqlens" in kwargs and _flash_attn_available:
            output = self._flash_attn_forward(hidden_states, position_ids, kwargs["cu_seqlens"], kwargs["max_seqlen"])
        else:
            output = self._eager_forward(hidden_states, attention_mask, position_ids, kv_cache)
        return output, None, None


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        attn_type = config.attn_types[layer_idx]
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx, attn_type=attn_type)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None, **kwargs):
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states_norm, attention_mask, position_ids, kv_cache, **kwargs
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs_embeds, padding_mask=None, attention_mask=None, position_ids=None, kv_cache=None):
        hidden_states = inputs_embeds * (self.config.hidden_size ** 0.5)

        # Determine if we should use Flash Attention
        use_flash = (
            _flash_attn_available and
            hidden_states.device.type == "cuda" and
            (kv_cache is None or kv_cache.num_items() == 0) and
            hidden_states.dtype in (torch.float16, torch.bfloat16) and
            padding_mask is not None
        )

        flash_kwargs = {}
        original_shape = hidden_states.shape
        
        if use_flash:
            # Prepare for Flash Attention variable length
            mask_bool = padding_mask.bool()
            hidden_states = hidden_states[mask_bool]
            
            # Ensure position_ids is properly handled and converted to int32
            if position_ids is not None:
                position_ids = position_ids[mask_bool].to(torch.int32)
            else:
                # Create position ids if not provided
                batch_size, seq_len = original_shape[:2]
                position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.int32)
                position_ids = position_ids.expand(batch_size, -1)[mask_bool]

            # Calculate sequence lengths and cumulative sequence lengths
            seqlens = padding_mask.sum(dim=1).to(dtype=torch.int32, device=hidden_states.device)
            cu_seqlens = torch.cat([
            torch.zeros(1, device=hidden_states.device, dtype=torch.int32),
            seqlens.cumsum(dim=0).to(dtype=torch.int32)  # <- ensure int32
        ])

            flash_kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": seqlens.max().item()}

        # Ensure position_ids is int32 for consistency
        elif position_ids is not None and position_ids.dtype != torch.int32:
            position_ids = position_ids.to(torch.int32)

        # Pass through decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, attention_mask, position_ids, kv_cache, **flash_kwargs
            )

        hidden_states = self.norm(hidden_states)

        # Restore original shape if using Flash Attention
        if use_flash:
            output = torch.zeros(original_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            output[padding_mask.bool()] = hidden_states
            hidden_states = output

        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, inputs_embeds, padding_mask, attention_mask, position_ids, kv_cache=None):
        
        hidden_states = self.model(inputs_embeds, padding_mask, attention_mask, position_ids, kv_cache)

        # If GemmaModel returns dict, unpack it
        if isinstance(hidden_states, dict):
            hidden_states, kv_cache = hidden_states["hidden_states"], hidden_states.get("kv_cache", kv_cache)

        logits = self.lm_head(hidden_states).float()

        return_data = {"logits": logits}
        if kv_cache is not None:
            
            return_data["kv_cache"] = kv_cache
        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_features):
        return self.linear(image_features)


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)
        final_embedding = inputs_embeds.clone()
        image_mask = input_ids == self.config.image_token_index
        
        # Ensure image_features has a batch dimension, even if it's 1
        if image_features.ndim == 2:
            image_features = image_features.unsqueeze(0)

        # Batch-wise replacement
        for i in range(final_embedding.shape[0]):
            mask_i = image_mask[i]
            if mask_i.sum() > 0:
                final_embedding[i, mask_i] = scaled_image_features[i]
                
        return final_embedding

    def forward(self, input_ids, pixel_values, attention_mask, kv_cache=None):
        
        # 1. Token embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # 2. Vision tower -> image features
        image_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        # 3. Project to LM hidden dim
        projected_features = self.multi_modal_projector(image_features)

        # 4. Merge text + image embeddings
        final_embeds = self._merge_input_ids_with_image_features(
            projected_features, inputs_embeds, input_ids
        )
        
        if attention_mask is None:
            
            
            
            
            # If no mask provided, assume everything is valid
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=inputs_embeds.device)[:, 0]
            # the line above yields shape [B]; expand below will make it [B, L]
        if attention_mask.dim() == 1:
            
            
            seq_len = input_ids.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len).contiguous()

        padding_mask = attention_mask

        batch_size, seq_len = input_ids.shape

        # 5. Build full attention mask (if not using FlashAttention)
        if padding_mask.shape[1] != input_ids.shape[1]:
            
            
            if padding_mask.shape[1] > input_ids.shape[1]:
                
                # Slice down (common case in autoregressive decoding)
                padding_mask = padding_mask[:, :input_ids.shape[1]]
            else:
                
                # Rare case: input_ids longer than padding_mask → pad with ones
                pad_len = input_ids.shape[1] - padding_mask.shape[1]
                pad = torch.ones(
                    (padding_mask.size(0), pad_len),
                    dtype=padding_mask.dtype,
                    device=padding_mask.device,
                )
                padding_mask = torch.cat([padding_mask, pad], dim=1)
        print("SYNCED: padding_mask", padding_mask.shape, "input_ids", input_ids.shape)
        
        final_attention_mask = None
        if kv_cache is None or kv_cache.num_items() == 0:
            dtype = inputs_embeds.dtype
            device = inputs_embeds.device
            print("MODEL DEBUG: padding_mask.shape:", padding_mask.shape, "input_ids.shape:", input_ids.shape)


            final_attention_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len, dtype=dtype, device=device
            )

            # (a) Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1
            )
            final_attention_mask.masked_fill_(
                causal_mask[None, None, :, :], torch.finfo(dtype).min
            )

            # (b) Padding mask (rows)
            padding_invalid = (padding_mask == 0)
            final_attention_mask.masked_fill_(
                padding_invalid[:, None, :, None], torch.finfo(dtype).min
            )

            # (c) Padding mask (cols)
            final_attention_mask.masked_fill_(
                padding_invalid[:, None, None, :], torch.finfo(dtype).min
            )

        # 6. 🚨 Actually run through the LM
        outputs = self.language_model(
            inputs_embeds=final_embeds,
            padding_mask=padding_mask,
            attention_mask=final_attention_mask,
            position_ids=None,   # supply if you have precomputed positions
            kv_cache=kv_cache,
        )

        # 7. Return logits + kv_cache dict
        return outputs
