from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from modeling import Adapter, CrossModelAdapter
from . import inference_mas as base
from long_context.compressor import PerceiverLatentCompressor
from long_context.datasets import build_synthetic_long_context_samples
from long_context.eval_utils import evaluate_long_context_predictions
from long_context.teacher_reader import (
    extract_teacher_hidden_states_for_sample,
    maybe_save_teacher_hidden_cache,
)

LONG_CONTEXT_LATENT_SLOT = "<<LONG_CONTEXT_LATENT_SLOT>>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental long-context distillation scaffold.")
    parser.add_argument("--dataset", type=str, default="synthetic", help="synthetic or path to jsonl")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--learner_model_name_or_path", type=str, required=True)
    parser.add_argument("--outer_el_path", type=str, default="")
    parser.add_argument("--outer_adapter_type_fallback", type=str, default="outer_ln_res_adapter", choices=["outer_ln_res_adapter"])

    parser.add_argument("--latent_inputs_path", type=str, default="", help=".pt/.npy file or directory for precomputed latents")
    parser.add_argument("--teacher_model_name_or_path", type=str, default="")
    parser.add_argument("--teacher_selected_layers", type=str, default="-1")
    parser.add_argument("--teacher_chunk_size", type=int, default=512)
    parser.add_argument("--teacher_chunk_overlap", type=int, default=64)
    parser.add_argument("--teacher_cache_dir", type=str, default="")
    parser.add_argument("--teacher_inner_aligner_path", type=str, default="")
    parser.add_argument("--inner_adapter_type_fallback", type=str, default="ln_res_adapter", choices=["ln_res_adapter"])

    parser.add_argument("--enable_compressor", type=int, default=0, choices=[0, 1])
    parser.add_argument("--compressor_num_latents", type=int, default=32)
    parser.add_argument("--compressor_num_layers", type=int, default=2)
    parser.add_argument("--compressor_num_heads", type=int, default=4)
    parser.add_argument("--compressor_latent_dim", type=int, default=0)
    parser.add_argument("--compressor_ckpt_path", type=str, default="")

    parser.add_argument("--run_long_context_distill", type=int, default=1, choices=[0, 1])
    parser.add_argument("--run_baseline_question", type=int, default=1, choices=[0, 1])
    parser.add_argument("--run_baseline_text_distill", type=int, default=1, choices=[0, 1])
    parser.add_argument("--expert_text_path", type=str, default="", help="Optional json/jsonl/text expert plans for text baseline")

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--dtype", type=str, default="auto", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--outer_dtype", type=str, default="auto", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--trust_remote_code", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--result_json", type=str, default="")
    return parser.parse_args()


def _parse_layers(layer_text: str) -> List[int]:
    out: List[int] = []
    for part in str(layer_text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or [-1]


def _load_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_samples(dataset: str, num_samples: int, seed: int) -> List[Dict[str, str]]:
    ds = str(dataset).strip().lower()
    if ds == "synthetic":
        return build_synthetic_long_context_samples(num_samples=num_samples, seed=seed)
    path = Path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path not found: {dataset}")
    if path.suffix.lower() == ".jsonl":
        rows = _load_jsonl(str(path))
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        raise ValueError("Dataset must be 'synthetic' or a .json/.jsonl file.")

    samples: List[Dict[str, str]] = []
    for i, row in enumerate(rows[: max(1, num_samples)]):
        if not isinstance(row, dict):
            continue
        samples.append(
            {
                "id": str(row.get("id", i)),
                "task_type": str(row.get("task_type", "unknown")),
                "context": str(row.get("context", "")),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
            }
        )
    if not samples:
        raise ValueError("No valid samples found in dataset file.")
    return samples


def _coerce_latent_tensor(x: object) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"Expected latent tensor [seq, dim], got shape={tuple(t.shape)}")
    return t.to(dtype=torch.float32)


def _unpack_latents_obj(obj: object) -> List[torch.Tensor]:
    if isinstance(obj, dict):
        if "latents" in obj:
            return _unpack_latents_obj(obj["latents"])
        if "latent" in obj:
            return _unpack_latents_obj(obj["latent"])
        raise ValueError("Unsupported latent dict format; expected key 'latents' or 'latent'.")
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 2:
            return [_coerce_latent_tensor(obj)]
        if obj.dim() == 3:
            return [_coerce_latent_tensor(obj[i]) for i in range(obj.size(0))]
        raise ValueError(f"Unsupported tensor latent shape {tuple(obj.shape)}")
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            return [_coerce_latent_tensor(obj)]
        if obj.ndim == 3:
            return [_coerce_latent_tensor(obj[i]) for i in range(obj.shape[0])]
        raise ValueError(f"Unsupported ndarray latent shape {obj.shape}")
    if isinstance(obj, list):
        return [_coerce_latent_tensor(x) for x in obj]
    raise ValueError(f"Unsupported latent object type: {type(obj)}")


def load_precomputed_latents(path: str, expected_n: int) -> List[torch.Tensor]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"latent path not found: {path}")

    latents: List[torch.Tensor]
    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.suffix.lower() in {".pt", ".npy"}])
        latents = []
        for p in files:
            if p.suffix.lower() == ".pt":
                obj = torch.load(p, map_location="cpu")
            else:
                obj = np.load(p, allow_pickle=True)
            items = _unpack_latents_obj(obj)
            latents.extend(items)
    else:
        if in_path.suffix.lower() == ".pt":
            obj = torch.load(in_path, map_location="cpu")
        elif in_path.suffix.lower() == ".npy":
            obj = np.load(in_path, allow_pickle=True)
        else:
            raise ValueError("latent_inputs_path must be .pt/.npy file or a directory")
        latents = _unpack_latents_obj(obj)

    if len(latents) == 1 and expected_n > 1:
        latents = [latents[0].clone() for _ in range(expected_n)]
    if len(latents) != expected_n:
        raise ValueError(f"Latent count mismatch: expected {expected_n}, got {len(latents)}")
    return latents


def _build_long_context_prompt_with_slot(context: str, question: str) -> str:
    return (
        "You are a learner model using distilled long-context latent guidance.\n"
        "Long-context latent signal:\n"
        f"{LONG_CONTEXT_LATENT_SLOT}\n"
        "Long context excerpt:\n"
        f"{context}\n"
        "Question:\n"
        f"{question}\n"
        "Provide only the final answer."
    )


def _build_text_distill_prompt(context: str, question: str, expert_text: str) -> str:
    return (
        "You are a learner model using a textual expert summary.\n"
        "Expert summary:\n"
        f"{expert_text}\n"
        "Long context excerpt:\n"
        f"{context}\n"
        "Question:\n"
        f"{question}\n"
        "Provide only the final answer."
    )


def _build_question_only_prompt(question: str) -> str:
    return f"Question:\n{question}\nProvide only the final answer."


def maybe_load_expert_texts(path: str, expected_n: int) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"expert_text_path not found: {path}")

    texts: List[str] = []
    if p.suffix.lower() == ".jsonl":
        for row in _load_jsonl(str(p)):
            texts.append(str(row.get("expert_output", row.get("text", ""))))
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            obj = obj.get("expert_outputs", obj.get("texts", []))
        if not isinstance(obj, list):
            raise ValueError("JSON expert_text_path must be a list or contain list key.")
        texts = [str(x.get("expert_output", x.get("text", "")) if isinstance(x, dict) else x) for x in obj]
    else:
        with open(p, "r", encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f]

    if len(texts) == 1 and expected_n > 1:
        texts = texts * expected_n
    if len(texts) != expected_n:
        raise ValueError(f"expert text count mismatch: expected {expected_n}, got {len(texts)}")
    return texts


def _infer_out_dim_from_file(path: str) -> int:
    return base.infer_outer_adapter_out_dim_from_file(path)


def map_long_context_latents_to_learner(
    latents: Sequence[torch.Tensor],
    outer_path: str,
    learner_hidden_dim: int,
    device: torch.device,
    outer_dtype: torch.dtype,
    outer_adapter_type: str,
) -> Tuple[List[torch.Tensor], CrossModelAdapter]:
    if not outer_path:
        raise ValueError("--outer_el_path is required for long-context latent mapping.")
    if not latents:
        raise ValueError("No latents to map.")

    in_dim = int(latents[0].size(-1))
    for t in latents:
        if int(t.size(-1)) != in_dim:
            raise ValueError("All latents must have same last dimension for outer adapter mapping.")
    out_dim = _infer_out_dim_from_file(outer_path)
    if out_dim != learner_hidden_dim:
        raise ValueError(f"Outer adapter output dim ({out_dim}) must equal learner hidden dim ({learner_hidden_dim}).")

    outer = base.load_outer_adapter_module(
        adapter_path=outer_path,
        in_dim=in_dim,
        out_dim=out_dim,
        adapter_type=outer_adapter_type,
        device=device,
        dtype=outer_dtype,
    )
    mapped = [base.run_outer_adapter(outer, t.to(device=device), output_dtype=torch.float32).detach().cpu() for t in latents]
    return mapped, outer


def generate_from_latents_with_slot(
    learner_model_name_or_path: str,
    samples: Sequence[Dict[str, str]],
    mapped_latents: Sequence[torch.Tensor],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    enable_thinking: bool,
) -> Tuple[List[str], List[int]]:
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=learner_model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name="long_context_learner",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype

    segments = [
        base.split_prompt_ids_by_slots(
            tokenizer,
            _build_long_context_prompt_with_slot(sample["context"], sample["question"]),
            [LONG_CONTEXT_LATENT_SLOT],
            enable_thinking,
        )
        for sample in samples
    ]

    gen_kwargs = base.build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    outputs: List[str] = []
    prompt_lengths: List[int] = []
    total_batches = (len(samples) + batch_size - 1) // batch_size
    for start, end in tqdm(base.batch_iter_indices(len(samples), batch_size), total=total_batches, desc="long_context_latent_generate"):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_prefix, seg_suffix = segments[idx]
            prefix = base.token_ids_to_embeds(embed_layer, seg_prefix, device=device, dtype=embed_dtype)
            suffix = base.token_ids_to_embeds(embed_layer, seg_suffix, device=device, dtype=embed_dtype)
            latent = mapped_latents[idx].to(device=device, dtype=embed_dtype)
            seq = torch.cat([prefix, latent, suffix], dim=0)
            embed_seqs.append(seq)

        batch_embeds, attention_mask = base.pad_left_embeds(embed_seqs, device=device)
        with torch.no_grad():
            generated = model.generate(inputs_embeds=batch_embeds, attention_mask=attention_mask, **gen_kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_len = attention_mask.size(1)
        gen_ids = sequences[:, prompt_len:] if sequences.size(1) > prompt_len else sequences
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend([t.strip() for t in texts])
        prompt_lengths.extend(attention_mask.sum(dim=1).tolist())

    base.release_resources(model, tokenizer)
    return outputs, [int(x) for x in prompt_lengths]


def run_question_only_baseline(
    learner_model_name_or_path: str,
    samples: Sequence[Dict[str, str]],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    enable_thinking: bool,
) -> List[str]:
    prompts = [_build_question_only_prompt(s["question"]) for s in samples]
    outputs, _ = base.run_text_generation_stage(
        stage_name="long_context_baseline_question",
        model_name_or_path=learner_model_name_or_path,
        user_prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enable_thinking=enable_thinking,
    )
    return outputs


def run_text_distill_baseline(
    learner_model_name_or_path: str,
    samples: Sequence[Dict[str, str]],
    expert_texts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    enable_thinking: bool,
) -> List[str]:
    prompts = [
        _build_text_distill_prompt(samples[i]["context"], samples[i]["question"], expert_texts[i])
        for i in range(len(samples))
    ]
    outputs, _ = base.run_text_generation_stage(
        stage_name="long_context_baseline_text_distill",
        model_name_or_path=learner_model_name_or_path,
        user_prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enable_thinking=enable_thinking,
    )
    return outputs


def _load_compressor(
    input_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> PerceiverLatentCompressor:
    latent_dim = args.compressor_latent_dim if args.compressor_latent_dim > 0 else input_dim
    compressor = PerceiverLatentCompressor(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_latents=args.compressor_num_latents,
        num_layers=args.compressor_num_layers,
        num_heads=args.compressor_num_heads,
    )
    if args.compressor_ckpt_path:
        state = torch.load(args.compressor_ckpt_path, map_location="cpu")
        compressor.load_state_dict(state, strict=True)
    compressor.to(device=device)
    compressor.eval()
    return compressor


def build_teacher_hidden_latents(
    samples: Sequence[Dict[str, str]],
    teacher_model_name_or_path: str,
    selected_layers: Sequence[int],
    chunk_size: int,
    chunk_overlap: int,
    teacher_cache_dir: str,
    teacher_inner_aligner_path: str,
    inner_adapter_type_fallback: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    enable_compressor: bool,
    compressor_args: argparse.Namespace,
) -> List[torch.Tensor]:
    teacher_model, teacher_tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=teacher_model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name="long_context_teacher",
    )

    maybe_inner: Optional[Adapter] = None
    if teacher_inner_aligner_path:
        hidden_size = teacher_model.get_input_embeddings().weight.size(-1)
        maybe_inner = base.load_inner_adapter_module(
            adapter_path=teacher_inner_aligner_path,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            fallback_adapter_type=inner_adapter_type_fallback,
        )

    raw_latents: List[torch.Tensor] = []
    for i, sample in enumerate(tqdm(samples, desc="teacher_hidden_extract")):
        hidden = extract_teacher_hidden_states_for_sample(
            model=teacher_model,
            tokenizer=teacher_tokenizer,
            context=sample["context"],
            question=sample["question"],
            device=device,
            selected_layers=selected_layers,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if maybe_inner is not None and hidden.numel() > 0:
            hidden = base.run_inner_adapter(maybe_inner, hidden.to(device=device), output_dtype=torch.float32).detach().cpu()

        if teacher_cache_dir:
            cache_path = str(Path(teacher_cache_dir) / f"sample_{i}.pt")
            maybe_save_teacher_hidden_cache(hidden, cache_path)
        raw_latents.append(hidden)

    if enable_compressor:
        input_dim = int(raw_latents[0].size(-1)) if raw_latents and raw_latents[0].numel() > 0 else teacher_model.get_input_embeddings().weight.size(-1)
        compressor = _load_compressor(input_dim=input_dim, args=compressor_args, device=device)
        compressed: List[torch.Tensor] = []
        for latent in tqdm(raw_latents, desc="compress_teacher_latents"):
            if latent.numel() == 0:
                compressed.append(latent)
                continue
            x = latent.to(device=device).unsqueeze(0)
            with torch.no_grad():
                y = compressor(x)
            compressed.append(y[0].detach().cpu().to(dtype=torch.float32))
        base.release_resources(compressor)
        raw_latents = compressed

    base.release_resources(teacher_model, teacher_tokenizer, maybe_inner)
    return raw_latents


def build_supervised_ce_training_tensors(
    embed_layer,
    prefix_ids: Sequence[int],
    latent_embeds: torch.Tensor,
    suffix_ids: Sequence[int],
    answer_ids: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training scaffold helper with correct full-answer teacher-forcing layout.

    inputs_embeds = prefix + latent + suffix + answer[:-1]
    labels = -100 for prefix/latent/suffix and answer token IDs for full answer segment.
    """
    if len(answer_ids) <= 0:
        raise ValueError("answer_ids must be non-empty")

    prefix = base.token_ids_to_embeds(embed_layer, prefix_ids, device=device, dtype=dtype)
    suffix = base.token_ids_to_embeds(embed_layer, suffix_ids, device=device, dtype=dtype)
    answer_input_ids = list(answer_ids[:-1])
    answer_target_ids = list(answer_ids)
    answer_in = base.token_ids_to_embeds(embed_layer, answer_input_ids, device=device, dtype=dtype)

    inputs_embeds = torch.cat([prefix, latent_embeds.to(device=device, dtype=dtype), suffix, answer_in], dim=0).unsqueeze(0)

    num_prefix = prefix.size(0)
    num_latent = latent_embeds.size(0)
    num_suffix = suffix.size(0)
    ignore = [-100] * (num_prefix + num_latent + num_suffix)
    labels = torch.tensor(ignore + answer_target_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, inputs_embeds.size(1)), dtype=torch.long, device=device)
    return inputs_embeds, attention_mask, labels


def main() -> None:
    args = parse_args()
    base._GEN_TOP_K = None
    base._GEN_MIN_P = None
    base._GEN_REPETITION_PENALTY = 1.0

    samples = load_samples(dataset=args.dataset, num_samples=args.num_samples, seed=args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = base.resolve_dtype(args.dtype)
    outer_dtype = base.resolve_dtype(args.outer_dtype)
    if model_dtype is None or outer_dtype is None:
        raise ValueError("Unsupported dtype configuration.")
    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        model_dtype = torch.float32
    if device.type == "cpu" and outer_dtype in {torch.float16, torch.bfloat16}:
        outer_dtype = torch.float32

    trust_remote_code = bool(args.trust_remote_code)
    enable_thinking = bool(args.enable_thinking)

    results: Dict[str, Dict[str, float]] = {}

    if bool(args.run_baseline_question):
        baseline_outputs = run_question_only_baseline(
            learner_model_name_or_path=args.learner_model_name_or_path,
            samples=samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
        )
        results["baseline_question"] = evaluate_long_context_predictions(samples, baseline_outputs)

    if bool(args.run_baseline_text_distill):
        expert_texts = maybe_load_expert_texts(args.expert_text_path, expected_n=len(samples))
        if expert_texts is not None:
            text_outputs = run_text_distill_baseline(
                learner_model_name_or_path=args.learner_model_name_or_path,
                samples=samples,
                expert_texts=expert_texts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                dtype=model_dtype,
                trust_remote_code=trust_remote_code,
                enable_thinking=enable_thinking,
            )
            results["baseline_text_distill"] = evaluate_long_context_predictions(samples, text_outputs)

    if bool(args.run_long_context_distill):
        if args.latent_inputs_path:
            teacher_latents = load_precomputed_latents(args.latent_inputs_path, expected_n=len(samples))
        else:
            if not args.teacher_model_name_or_path:
                raise ValueError("Provide --latent_inputs_path or --teacher_model_name_or_path for long-context distillation path.")
            teacher_latents = build_teacher_hidden_latents(
                samples=samples,
                teacher_model_name_or_path=args.teacher_model_name_or_path,
                selected_layers=_parse_layers(args.teacher_selected_layers),
                chunk_size=args.teacher_chunk_size,
                chunk_overlap=args.teacher_chunk_overlap,
                teacher_cache_dir=args.teacher_cache_dir,
                teacher_inner_aligner_path=args.teacher_inner_aligner_path,
                inner_adapter_type_fallback=args.inner_adapter_type_fallback,
                device=device,
                dtype=model_dtype,
                trust_remote_code=trust_remote_code,
                enable_compressor=bool(args.enable_compressor),
                compressor_args=args,
            )

        learner_model, learner_tokenizer = base.load_agent_model_and_tokenizer(
            model_name_or_path=args.learner_model_name_or_path,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            agent_name="long_context_learner_shape_probe",
        )
        learner_hidden = learner_model.get_input_embeddings().weight.size(-1)
        base.release_resources(learner_model, learner_tokenizer)

        mapped_latents, outer_module = map_long_context_latents_to_learner(
            latents=teacher_latents,
            outer_path=args.outer_el_path,
            learner_hidden_dim=learner_hidden,
            device=device,
            outer_dtype=outer_dtype,
            outer_adapter_type=args.outer_adapter_type_fallback,
        )

        outputs, prompt_lengths = generate_from_latents_with_slot(
            learner_model_name_or_path=args.learner_model_name_or_path,
            samples=samples,
            mapped_latents=mapped_latents,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
        )
        results["long_context_distillation"] = evaluate_long_context_predictions(
            samples,
            outputs,
            latent_lengths=[int(x.size(0)) for x in mapped_latents],
            prompt_token_lengths=prompt_lengths,
        )
        base.release_resources(outer_module)

    print("[long_context_distill] metrics")
    for name, metrics in results.items():
        joined = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"- {name}: {joined}")

    if args.result_json.strip():
        out_path = Path(args.result_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "num_samples": len(samples),
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
