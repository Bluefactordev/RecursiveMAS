from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch


def _chunk_tokens(token_ids: List[int], chunk_size: int, overlap: int) -> Iterable[List[int]]:
    step = max(1, chunk_size - max(0, overlap))
    for i in range(0, len(token_ids), step):
        chunk = token_ids[i : i + chunk_size]
        if chunk:
            yield chunk
        if i + chunk_size >= len(token_ids):
            break


@torch.no_grad()
def extract_teacher_hidden_states_for_sample(
    model,
    tokenizer,
    context: str,
    question: str,
    device: torch.device,
    selected_layers: Sequence[int],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> torch.Tensor:
    if not selected_layers:
        raise ValueError("selected_layers cannot be empty.")
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    question_ids = tokenizer("\nQuestion:\n" + question, add_special_tokens=False)["input_ids"]

    rows: List[torch.Tensor] = []
    for chunk_ids in _chunk_tokens(context_ids, chunk_size=chunk_size, overlap=chunk_overlap):
        combined = chunk_ids + question_ids
        input_ids = torch.tensor(combined, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        for layer_idx in selected_layers:
            layer_hidden = outputs.hidden_states[layer_idx][0]
            pooled = layer_hidden.mean(dim=0)
            rows.append(pooled)

    if not rows:
        hidden_size = model.get_input_embeddings().weight.size(-1)
        return torch.empty((0, hidden_size), dtype=torch.float32)
    return torch.stack(rows, dim=0).to(dtype=torch.float32)


def maybe_save_teacher_hidden_cache(hidden_states: torch.Tensor, output_path: Optional[str]) -> None:
    if not output_path:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hidden_states.detach().cpu(), out)
