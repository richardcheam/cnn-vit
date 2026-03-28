from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    """Feed-forward sublayer used inside each transformer encoder block."""
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm self-attention block with a standard residual MLP."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        # Returning attention is optional because we only need those tensors for
        # interpretability, not for ordinary training.
        normalized = self.norm1(x)
        attended, attention_weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout(attended)
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            return x, attention_weights
        return x


class VisionTransformer(nn.Module):
    """Compact ViT tailored to 32x32 CIFAR-10 images."""
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        # The convolution acts as the patch extractor and linear projection in
        # one operation, producing one token per non-overlapping image patch.
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + self.positional_embedding[:, : tokens.size(1)]
        return self.dropout(tokens)

    def forward_features(
        self,
        x: torch.Tensor,
        return_attentions: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # `tokens[:, 0]` is the class token, while the remaining tokens represent
        # spatial patches. We keep the full sequence for attention visualization.
        tokens = self._embed_patches(x)
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            if return_attentions:
                tokens, attention = block(tokens, return_attention=True)
                attentions.append(attention)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens, attentions

    def forward(self, x: torch.Tensor, return_attentions: bool = False):
        tokens, attentions = self.forward_features(x, return_attentions=return_attentions)
        # Classification is done from the class token after the encoder stack.
        logits = self.head(tokens[:, 0])
        if return_attentions:
            return logits, attentions, tokens
        return logits

    @property
    def patch_grid_shape(self) -> tuple[int, int]:
        side = int(math.sqrt(self.num_patches))
        return side, side
