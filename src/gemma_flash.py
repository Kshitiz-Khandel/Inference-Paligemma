import torch
from torch import nn
from typing import Optional, Tuple, List
import math
import enum
from siglip import SiglipVisionConfig, SiglipVisionModel
import torch.nn.functional as F

# Flash Attention import
try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
    print("FlashAttention is available and will be used.")
except ImportError:
    print("FlashAttention is not installed. Falling back to default PyTorch attention.")
    _flash_attn_available = False


class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
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
        super().__init__()
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
        
        # Default to alternating attention types if not specified
        if attn_types is None:
            assert self.num_hidden_layers % 2 == 0
            self.attn_types = [AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * int(self.num_hidden_layers/2)
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
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

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
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
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
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = torch.einsum(
                "d,bs->bsd",
                self.inv_freq.to(x.device).float(),
                position_ids.float(),
            )
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
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


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
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.sliding_window_size = config.sliding_window_size
        assert self.hidden_size % self.num_heads == 0        

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        is_decode_phase = (q_len == 1) and (kv_cache is not None and kv_cache.num_items() > 0)
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
    
        query_states = torch.einsum('bsnh->bnsh', query_states)
        key_states = torch.einsum('bsnh->bnsh', key_states)
        value_states = torch.einsum('bsnh->bnsh', value_states)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        final_mask = attention_mask
        if self.attn_type == AttentionType.LOCAL_SLIDING and self.sliding_window_size is not None:
            seq_len = attention_mask.shape[-1]
            all_ones = torch.ones_like(attention_mask)
            sliding_mask = torch.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * torch.tril(all_ones, self.sliding_window_size - 1)
            
            dtype = attention_mask.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = torch.where(sliding_mask == 1, attention_mask, min_dtype)
            final_mask = attention_mask
        
        use_flash_attn = (
            _flash_attn_available and 
            not is_decode_phase and 
            hidden_states.device.type == "cuda" and
            hidden_states.dtype in (torch.float16, torch.bfloat16) and
            final_mask is not None and final_mask.dim() == 4
        )
        
        if use_flash_attn:
            print(f"Layer {self.layer_idx}: Using Flash Attention (Prefill)")
            q_flash = query_states.transpose(1, 2)
            k_flash = key_states.transpose(1, 2)
            v_flash = value_states.transpose(1, 2)
            
            attn_output = flash_attn_func(
                q_flash, k_flash, v_flash,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=(self.sliding_window_size - 1, 0) if self.attn_type == AttentionType.LOCAL_SLIDING else (-1, -1)
            )
            
        else:
            print(f"Layer {self.layer_idx}: Using SDPA ({'Decode' if is_decode_phase else 'Prefill'})")
            attn_output = F.scaled_dot_product_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                attn_mask=final_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            
            attn_output = attn_output.transpose(1, 2).contiguous()
    
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
    
        return attn_output, None


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        attn_type = config.attn_types[layer_idx] if config.attn_types and layer_idx < len(config.attn_types) else AttentionType.GLOBAL
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx, attn_type=attn_type)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return_data = {"logits": logits}
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        batch_size, sequence_length, embed_dim = inputs_embeds.shape
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        text_mask_expanded = text_mask.unsqueeze(-1)
        image_mask_expanded = image_mask.unsqueeze(-1)
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        #### CREATE THE ATTENTION MASK ####
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase
            causal_mask = torch.full((q_len, q_len), fill_value=0, dtype=dtype, device=device)
            causal_mask = causal_mask.triu(diagonal=1) * min_dtype
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Decoding phase
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.zeros((batch_size, q_len, kv_len), dtype=dtype, device=device)

        input_attention_mask_expanded = (1.0 - attention_mask[:, None, None, :]).to(dtype) * min_dtype
        causal_mask = causal_mask.unsqueeze(1) + input_attention_mask_expanded

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.sum(dim=-1, keepdim=True) - 1
            position_ids = torch.clamp(position_ids, min=0)
        else:
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(attention_mask == 0, 0)
            position_ids.clamp_(min=0)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
            image_features = self.multi_modal_projector(selected_image_feature)

            inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        else:
            # Decode phase logic when no new images are processed
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            min_dtype = torch.finfo(dtype).min
            batch_size, q_len = input_ids.shape
            
            if kv_cache is not None and kv_cache.num_items() > 0:
                kv_len = kv_cache.num_items() + q_len
                causal_mask = torch.zeros((batch_size, q_len, kv_len), dtype=dtype, device=device)
                position_ids = attention_mask.sum(dim=-1, keepdim=True) - 1
                position_ids = torch.clamp(position_ids, min=0)
            else:
                causal_mask = torch.full((q_len, q_len), fill_value=0, dtype=dtype, device=device)
                causal_mask = causal_mask.triu(diagonal=1) * min_dtype
                causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
                position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(attention_mask == 0, 0)
                position_ids.clamp_(min=0)
            
            input_attention_mask_expanded = (1.0 - attention_mask[:, None, None, :]).to(dtype) * min_dtype
            attention_mask = causal_mask.unsqueeze(1) + input_attention_mask_expanded
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs


    
    
 