from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size, channels, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        patch_embeds = self.patch_embedding(pixel_values)  
        
        # Use einsum for more efficient flattening and transposition
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Num_Patches, Embed_Dim]
        num_patches_h, num_patches_w = patch_embeds.shape[2], patch_embeds.shape[3]
        embeddings = torch.einsum('bchw->bwhc', patch_embeds).reshape(batch_size, num_patches_h * num_patches_w, self.embed_dim)
        
        # Use broadcasting for position embeddings addition - more memory efficient
        # position_ids: [1, Num_Patches] -> position_embedding: [1, Num_Patches, Embed_Dim]
        # embeddings: [Batch_Size, Num_Patches, Embed_Dim] + [1, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        position_embeds = self.position_embedding(self.position_ids)  # [1, Num_Patches, Embed_Dim]
        embeddings = embeddings + position_embeds  # Broadcasting across batch dimension
        
        return embeddings


class SiglipAttention(nn.Module):
    """Multi-headed attention with einsum optimizations"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to multi-head format: [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [Batch_Size, Num_Heads, Num_Patches, Head_Dim] using einsum
        query_states = torch.einsum('bshd->bhsd', query_states)
        key_states = torch.einsum('bshd->bhsd', key_states)
        value_states = torch.einsum('bshd->bhsd', value_states)
        
        # Compute attention scores using einsum - more efficient than matmul + transpose
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        # key_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        # attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states, key_states) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise with better numerical stability
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout only during training
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention weights to values using einsum
        # attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        # value_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        # attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Reshape back to original dimensions using einsum
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = torch.einsum('bhsd->bshd', attn_output).reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        
        # Use F.gelu for better performance than nn.functional.gelu
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # Pre-norm architecture with residual connections
        # First residual connection around attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        
        # Use in-place addition for memory efficiency where possible
        hidden_states = hidden_states + residual
        
        # Second residual connection around MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        # Sequential processing through encoder layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        # Process through transformer encoder
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        # Final layer normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """Optimized SigLIP Vision Model with einsum operations and broadcasting"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> torch.Tensor:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)





