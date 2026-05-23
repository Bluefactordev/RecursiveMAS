from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PerceiverLatentCompressor(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_latents: int = 32, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.ffn = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim * 4), nn.GELU(), nn.Linear(latent_dim * 4, latent_dim)) for _ in range(num_layers)]
        )
        self.out_ln = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected input shape [batch, seq, dim].")
        h = self.input_proj(x)
        latents = self.latents.unsqueeze(0).expand(h.size(0), -1, -1)
        for attn, ffn in zip(self.cross_attn, self.ffn):
            attn_out, _ = attn(query=latents, key=h, value=h, key_padding_mask=key_padding_mask)
            latents = latents + attn_out
            latents = latents + ffn(latents)
        return self.out_ln(latents)
