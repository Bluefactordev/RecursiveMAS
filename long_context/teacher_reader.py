from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm

from inference_utils import inference_mas as base


def parse_layer_indices(layer_spec: str) -> List[int]:
    if not layer_spec.strip():
        return [-1]
    return [int(part.strip()) for part in layer_spec.split(",") if part.strip()]


def _chunk_ranges(total_length: int, chunk_tokens: int, overlap_tokens: int) -> Iterable[tuple[int, int]]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    step = chunk_tokens - overlap_tokens
    start = 0
    while start < total_length:
        end = min(total_length, start + chunk_tokens)
        yield start, end
        if end >= total_length:
            break
        start += step


def build_teacher_reader_prompt(context_chunk: str, question: str) -> str:
    context_text = str(context_chunk).strip()
    question_text = str(question).strip()
    if question_text:
        return (
            "Read the context chunk and preserve information relevant to the downstream question.\n"
            "Context chunk:\n"
            f"{context_text}\n\n"
            "Question:\n"
            f"{question_text}"
        )
    return (
        "Read the context chunk and preserve information for later reasoning.\n"
        "Context chunk:\n"
        f"{context_text}"
    )


@torch.no_grad()
def extract_teacher_hidden_state_cache(
    *,
    model_name_or_path: str,
    samples: Sequence[Dict[str, Any]],
    layer_indices: Sequence[int],
    chunk_tokens: int,
    overlap_tokens: int,
    device: torch.device,
    dtype: torch.dtype | str,
    trust_remote_code: bool,
    enable_thinking: bool,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name="long_context_teacher",
    )
    cache_dir = Path(output_dir).resolve() if output_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_name_or_path": model_name_or_path,
            "layer_indices": list(layer_indices),
            "chunk_tokens": int(chunk_tokens),
            "overlap_tokens": int(overlap_tokens),
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_records: List[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(tqdm(samples, desc="teacher_hidden_cache")):
        context = str(sample.get("context", ""))
        question = str(sample.get("question", ""))
        token_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        chunk_records: List[Dict[str, Any]] = []
        for chunk_idx, (start, end) in enumerate(_chunk_ranges(len(token_ids), chunk_tokens, overlap_tokens)):
            chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
            prompt_ids = base.render_chat_prompt_ids(
                tokenizer,
                build_teacher_reader_prompt(chunk_text, question),
                enable_thinking=enable_thinking,
            )
            input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            selected = [outputs.hidden_states[layer_idx][0].detach().cpu() for layer_idx in layer_indices]
            chunk_records.append(
                {
                    "chunk_index": chunk_idx,
                    "token_start": int(start),
                    "token_end": int(end),
                    "hidden_states": torch.stack(selected, dim=0),
                }
            )
        record = {
            "sample_id": sample.get("sample_id", sample_idx),
            "question": question,
            "context": context,
            "task": sample.get("task", "custom"),
            "layer_indices": list(layer_indices),
            "chunks": chunk_records,
        }
        if cache_dir is not None:
            torch.save(record, cache_dir / f"sample_{sample_idx:05d}.pt")
        all_records.append(record)

    base.release_resources(model, tokenizer)
    return all_records


def load_teacher_hidden_state_cache(cache_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(cache_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Teacher cache directory not found: {path}")
    files = sorted(path.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No teacher cache files found in: {path}")
    return [torch.load(file_path, map_location="cpu") for file_path in files]


def teacher_cache_to_latents(
    cache_record: Dict[str, Any],
    *,
    layer_reduce: str = "mean",
    token_pooling: str = "last_token",
) -> torch.Tensor:
    chunk_latents: List[torch.Tensor] = []
    for chunk in cache_record.get("chunks", []):
        hidden_states = chunk["hidden_states"]
        if layer_reduce == "mean":
            reduced = hidden_states.mean(dim=0)
        elif layer_reduce == "last":
            reduced = hidden_states[-1]
        else:
            raise ValueError(f"Unsupported layer_reduce: {layer_reduce}")
        if token_pooling == "last_token":
            reduced = reduced[-1:, :]
        elif token_pooling == "mean":
            reduced = reduced.mean(dim=0, keepdim=True)
        elif token_pooling == "concat":
            reduced = reduced
        else:
            raise ValueError(f"Unsupported token_pooling: {token_pooling}")
        chunk_latents.append(reduced)
    if not chunk_latents:
        raise ValueError("Teacher cache record contains no chunks")
    return torch.cat(chunk_latents, dim=0).to(torch.float32)
