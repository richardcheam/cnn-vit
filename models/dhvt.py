from __future__ import annotations

import math

import torch
from torch import nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth applied per sample on residual branches."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


class Affine(nn.Module):
    """Channel-wise affine transform used in the official DHVT patch embed."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta


class ConvPatchEmbed(nn.Module):
    """Convolutional patch embedding (SOPE) from the official DHVT code."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        if patch_size == 16:
            self.proj = nn.Sequential(
                conv3x3(in_channels, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 4:
            self.proj = nn.Sequential(
                conv3x3(in_channels, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 2:
            self.proj = nn.Sequential(
                conv3x3(in_channels, embed_dim, 2),
                nn.GELU(),
            )
        else:
            raise ValueError("DHVT supports patch_size in {2, 4, 16}.")

        self.pre_affine = Affine(in_channels)
        self.post_affine = Affine(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_affine(x)
        x = self.proj(x)
        x = self.post_affine(x)
        return x.flatten(2).transpose(1, 2)


class DAFF(nn.Module):
    """Dynamic adaptive feed-forward block from the official DHVT code."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=hidden_features,
        )
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_features, in_features // 4)
        self.excitation = nn.Linear(in_features // 4, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        cls_token, patch_tokens = torch.split(x, [1, num_tokens - 1], dim=1)
        grid_size = int(math.sqrt(num_tokens - 1))
        patch_tokens = (
            patch_tokens.reshape(batch_size, grid_size, grid_size, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        patch_tokens = self.conv1(patch_tokens)
        patch_tokens = self.bn1(patch_tokens)
        patch_tokens = self.act(patch_tokens)

        shortcut = patch_tokens
        patch_tokens = self.conv2(patch_tokens)
        patch_tokens = self.bn2(patch_tokens)
        patch_tokens = self.act(patch_tokens)
        patch_tokens = shortcut + patch_tokens

        patch_tokens = self.conv3(patch_tokens)
        patch_tokens = self.bn3(patch_tokens)
        patch_tokens = self.drop(patch_tokens)

        weight = self.squeeze(patch_tokens).flatten(1).reshape(batch_size, 1, channels)
        weight = self.excitation(self.act(self.compress(weight)))
        cls_token = cls_token * weight

        patch_tokens = patch_tokens.flatten(2).permute(0, 2, 1).contiguous()
        return torch.cat((cls_token, patch_tokens), dim=1)


class HIAttention(nn.Module):
    """Head-token interaction attention (HI-MHSA) from the official DHVT code."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim // num_heads, dim, bias=True)
        self.ht_norm = nn.LayerNorm(dim // num_heads)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_heads, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        batch_size, num_tokens, channels = x.shape

        # Generate one head token per attention head from channel groups.
        head_pos = self.pos_embed.expand(batch_size, -1, -1)
        grouped = x.reshape(batch_size, num_tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        grouped = grouped.mean(dim=2)
        grouped = self.ht_proj(grouped).reshape(batch_size, -1, self.num_heads, channels // self.num_heads)
        grouped = self.act(self.ht_norm(grouped)).flatten(2)
        grouped = grouped + head_pos
        x = torch.cat([x, grouped], dim=1)

        total_tokens = x.shape[1]
        qkv = self.qkv(x).reshape(batch_size, total_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(batch_size, total_tokens, channels)
        x = self.proj(x)

        cls_token, patch_tokens, head_tokens = torch.split(
            x,
            [1, num_tokens - 1, self.num_heads],
            dim=1,
        )
        cls_token = cls_token + head_tokens.mean(dim=1, keepdim=True)
        x = torch.cat([cls_token, patch_tokens], dim=1)
        x = self.proj_drop(x)

        if return_attention:
            # Approximate rollout support: keep only the original cls+patch token
            # submatrix so downstream attention visualization stays compatible.
            reduced_attention = attention[:, :, :num_tokens, :num_tokens]
            return x, reduced_attention
        return x


class DHVTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HIAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = DAFF(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            kernel_size=3,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if return_attention:
            attended, attention = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attended)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attention

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DHVisionTransformer(nn.Module):
    """Compact DHVT port adapted from the official NeurIPS 2022 implementation."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size
        self.embed_dim = embed_dim

        self.patch_embed = ConvPatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        drop_path_schedule = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                DHVTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=dropout,
                    attn_drop=attention_dropout,
                    drop_path_rate=drop_path_schedule[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(
        self,
        x: torch.Tensor,
        return_attentions: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        patch_tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = self.pos_drop(tokens)

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
        logits = self.head(tokens[:, 0])
        if return_attentions:
            return logits, attentions, tokens
        return logits

    @property
    def patch_grid_shape(self) -> tuple[int, int]:
        return self.grid_size, self.grid_size
