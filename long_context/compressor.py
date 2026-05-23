from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class _PerceiverBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ln_attn = nn.LayerNorm(dim)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, latents: torch.Tensor, source: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=self.ln_attn(latents),
            key=source,
            value=source,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = latents + attn_out
        latents = latents + self.mlp(self.ln_mlp(latents))
        return latents


class PerceiverLatentCompressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        num_latents: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.num_latents = int(num_latents)
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.input_dim) / (self.input_dim ** 0.5))
        self.blocks = nn.ModuleList(
            _PerceiverBlock(self.input_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        )
        self.final_ln = nn.LayerNorm(self.input_dim)

    def forward(self, source: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if source.dim() != 3:
            raise ValueError(f"Expected source to have shape [batch, seq, dim], got {tuple(source.shape)}")
        if source.size(-1) != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {source.size(-1)}")
        latents = self.latents.unsqueeze(0).expand(source.size(0), -1, -1)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.to(dtype=torch.bool, device=source.device)
            key_padding_mask = ~key_padding_mask
        for block in self.blocks:
            latents = block(latents, source, key_padding_mask)
        return self.final_ln(latents)


def load_perceiver_compressor(
    path: str,
    *,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype | str,
    num_latents: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
) -> PerceiverLatentCompressor:
    payload: Dict[str, Any] | Dict[str, torch.Tensor] = torch.load(path, map_location="cpu")
    config: Dict[str, Any] = {}
    state_dict = payload
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        config = dict(payload.get("config", {}))
    compressor = PerceiverLatentCompressor(
        input_dim=input_dim,
        num_latents=int(config.get("num_latents", num_latents)),
        num_layers=int(config.get("num_layers", num_layers)),
        num_heads=int(config.get("num_heads", num_heads)),
    )
    compressor.load_state_dict(state_dict, strict=True)
    resolved_dtype = dtype
    if dtype == "auto":
        resolved_dtype = next(
            (value.dtype for value in state_dict.values() if torch.is_tensor(value) and value.is_floating_point()),
            torch.float32,
        )
    compressor.to(device=device, dtype=resolved_dtype)
    compressor.eval()
    for param in compressor.parameters():
        param.requires_grad = False
    return compressor
