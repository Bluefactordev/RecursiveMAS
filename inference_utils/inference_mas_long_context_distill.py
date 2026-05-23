from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from . import inference_mas as base
from long_context.compressor import load_perceiver_compressor
from long_context.datasets import (
    build_synthetic_long_context_dataset,
    load_long_context_samples,
    save_long_context_samples,
)
from long_context.eval_utils import compute_long_context_metrics
from long_context.teacher_reader import (
    extract_teacher_hidden_state_cache,
    load_teacher_hidden_state_cache,
    parse_layer_indices,
    teacher_cache_to_latents,
)


LONG_CONTEXT_LATENT_SLOT = "<<LONG_CONTEXT_LATENT_SLOT>>"
RUN_MODE_LONG_CONTEXT = "long_context_distillation"
RUN_MODE_QUESTION_ONLY = "question_only"
RUN_MODE_TEXT_DISTILL = "text_distill"
SUPPORTED_RUN_MODES = {RUN_MODE_LONG_CONTEXT, RUN_MODE_QUESTION_ONLY, RUN_MODE_TEXT_DISTILL}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental long-context distillation scaffold for RecursiveMAS.")
    parser.add_argument("--learner_model_name_or_path", type=str, default="")
    parser.add_argument("--outer_adapter_path", type=str, default="")
    parser.add_argument(
        "--run_modes",
        type=str,
        default=RUN_MODE_QUESTION_ONLY,
        help="Comma-separated list: long_context_distillation,question_only,text_distill",
    )
    parser.add_argument("--dataset_jsonl", type=str, default="")
    parser.add_argument("--synthetic_num_samples", type=int, default=8)
    parser.add_argument("--synthetic_seed", type=int, default=42)
    parser.add_argument("--synthetic_tasks", type=str, default="needle,latest_value,two_hop,symbolic_copy")
    parser.add_argument("--synthetic_filler_lines", type=int, default=24)
    parser.add_argument("--write_synthetic_dataset_path", type=str, default="")
    parser.add_argument("--long_context_latents_path", type=str, default="")
    parser.add_argument("--teacher_model_name_or_path", type=str, default="")
    parser.add_argument("--teacher_cache_dir", type=str, default="")
    parser.add_argument("--teacher_layers", type=str, default="-1")
    parser.add_argument("--teacher_chunk_tokens", type=int, default=512)
    parser.add_argument("--teacher_chunk_overlap_tokens", type=int, default=64)
    parser.add_argument("--teacher_layer_reduce", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--teacher_token_pooling", type=str, default="last_token", choices=["last_token", "mean", "concat"])
    parser.add_argument("--compressor_path", type=str, default="")
    parser.add_argument("--compressor_num_latents", type=int, default=32)
    parser.add_argument("--compressor_num_layers", type=int, default=2)
    parser.add_argument("--compressor_num_heads", type=int, default=4)
    parser.add_argument(
        "--outer_adapter_type_fallback",
        type=str,
        default="outer_ln_res_adapter",
        choices=["outer_ln_res_adapter"],
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=-1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="auto", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--outer_dtype", type=str, default="auto", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--trust_remote_code", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--result_jsonl", type=str, default="")
    parser.add_argument("--training_placeholder_path", type=str, default="")
    return parser.parse_args()


def parse_run_modes(run_modes: str) -> List[str]:
    modes = [mode.strip().lower() for mode in run_modes.split(",") if mode.strip()]
    if not modes:
        raise ValueError("At least one run mode is required.")
    for mode in modes:
        if mode not in SUPPORTED_RUN_MODES:
            raise ValueError(f"Unsupported run mode: {mode}")
    return modes


def load_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset_jsonl:
        return load_long_context_samples(args.dataset_jsonl)
    task_names = [part.strip() for part in args.synthetic_tasks.split(",") if part.strip()]
    samples = build_synthetic_long_context_dataset(
        args.synthetic_num_samples,
        seed=args.synthetic_seed,
        tasks=task_names,
        filler_lines=args.synthetic_filler_lines,
    )
    if args.write_synthetic_dataset_path:
        save_long_context_samples(args.write_synthetic_dataset_path, samples)
    return samples


def build_long_context_prompt_with_slot(question: str) -> str:
    return (
        "You are the learner agent in an experimental long-context distillation setup.\n"
        "Long-context latent evidence:\n"
        f"{LONG_CONTEXT_LATENT_SLOT}\n"
        "Use the latent evidence as a soft signal and answer the question directly.\n"
        "Question:\n"
        f"{question}"
    )


def build_question_only_prompt(question: str) -> str:
    return (
        "You are the learner agent.\n"
        "Answer the question directly.\n"
        "Question:\n"
        f"{question}"
    )


def build_text_distill_prompt(question: str, expert_text: str) -> str:
    return (
        "You are the learner agent in a distillation setup.\n"
        "Expert plan:\n"
        f"{expert_text}\n\n"
        "Question:\n"
        f"{question}\n"
        "Follow the plan and answer directly."
    )


def _coerce_latent_tensor(item: Any) -> torch.Tensor:
    if isinstance(item, torch.Tensor):
        tensor = item.detach().cpu().to(torch.float32)
    else:
        tensor = torch.as_tensor(item, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"Expected latent tensor with shape [seq, dim], got {tuple(tensor.shape)}")
    return tensor.contiguous()


def load_precomputed_latents(path: str, expected_count: Optional[int] = None) -> List[torch.Tensor]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Latent path not found: {input_path}")
    if input_path.is_dir():
        files = sorted(list(input_path.glob("*.pt")) + list(input_path.glob("*.npy")))
        if not files:
            raise FileNotFoundError(f"No .pt/.npy latent files found in: {input_path}")
        latents = [load_precomputed_latents(str(file_path), expected_count=None)[0] for file_path in files]
    elif input_path.suffix.lower() == ".pt":
        payload = torch.load(input_path, map_location="cpu")
        if isinstance(payload, dict) and "latents" in payload:
            payload = payload["latents"]
        if isinstance(payload, torch.Tensor) and payload.dim() == 3:
            latents = [_coerce_latent_tensor(payload[idx]) for idx in range(payload.size(0))]
        elif isinstance(payload, (list, tuple)):
            latents = [_coerce_latent_tensor(item) for item in payload]
        else:
            latents = [_coerce_latent_tensor(payload)]
    elif input_path.suffix.lower() == ".npy":
        payload = np.load(input_path, allow_pickle=True)
        if payload.dtype == object:
            latents = [_coerce_latent_tensor(item) for item in payload.tolist()]
        elif payload.ndim == 3:
            latents = [_coerce_latent_tensor(payload[idx]) for idx in range(payload.shape[0])]
        else:
            latents = [_coerce_latent_tensor(payload)]
    else:
        raise ValueError(f"Unsupported latent file type: {input_path.suffix}")
    if expected_count is not None and len(latents) != expected_count:
        raise ValueError(f"Expected {expected_count} latent samples, found {len(latents)}")
    return latents


@torch.no_grad()
def maybe_compress_latents(
    source_latents: Sequence[torch.Tensor],
    *,
    compressor_path: str,
    device: torch.device,
    dtype: torch.dtype | str,
    num_latents: int,
    num_layers: int,
    num_heads: int,
) -> List[torch.Tensor]:
    if not compressor_path:
        return [latent.to(torch.float32) for latent in source_latents]
    input_dim = int(source_latents[0].size(-1))
    compressor = load_perceiver_compressor(
        compressor_path,
        input_dim=input_dim,
        device=device,
        dtype=dtype,
        num_latents=num_latents,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    batches: List[torch.Tensor] = []
    for latent in source_latents:
        batches.append(latent)
    padded, attention_mask = base.pad_left_embeds([item.to(device=device, dtype=next(compressor.parameters()).dtype) for item in batches], device=device)
    compressed = compressor(padded, attention_mask=attention_mask)
    outputs = [compressed[idx].detach().cpu().to(torch.float32) for idx in range(compressed.size(0))]
    base.release_resources(compressor)
    return outputs


@torch.no_grad()
def build_source_latents(
    args: argparse.Namespace,
    samples: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    model_dtype: torch.dtype | str,
    trust_remote_code: bool,
    enable_thinking: bool,
) -> Optional[List[torch.Tensor]]:
    if args.long_context_latents_path:
        return load_precomputed_latents(args.long_context_latents_path, expected_count=len(samples))
    if not args.teacher_model_name_or_path:
        return None

    cache_records: List[Dict[str, Any]]
    if args.teacher_cache_dir and os.path.isdir(args.teacher_cache_dir) and list(Path(args.teacher_cache_dir).glob("sample_*.pt")):
        cache_records = load_teacher_hidden_state_cache(args.teacher_cache_dir)
    else:
        cache_records = extract_teacher_hidden_state_cache(
            model_name_or_path=args.teacher_model_name_or_path,
            samples=samples,
            layer_indices=parse_layer_indices(args.teacher_layers),
            chunk_tokens=args.teacher_chunk_tokens,
            overlap_tokens=args.teacher_chunk_overlap_tokens,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            output_dir=(args.teacher_cache_dir or None),
        )
    if len(cache_records) != len(samples):
        raise ValueError("Teacher cache/sample size mismatch")
    source_latents = [
        teacher_cache_to_latents(
            record,
            layer_reduce=args.teacher_layer_reduce,
            token_pooling=args.teacher_token_pooling,
        )
        for record in cache_records
    ]
    return maybe_compress_latents(
        source_latents,
        compressor_path=args.compressor_path,
        device=device,
        dtype=model_dtype,
        num_latents=args.compressor_num_latents,
        num_layers=args.compressor_num_layers,
        num_heads=args.compressor_num_heads,
    )


def run_question_only_baseline(
    *,
    learner_model_name_or_path: str,
    questions: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype | str,
    trust_remote_code: bool,
    enable_thinking: bool,
) -> List[str]:
    prompts = [build_question_only_prompt(question) for question in questions]
    outputs, _ = base.run_text_generation_stage(
        stage_name="long_context_question_only",
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
    *,
    learner_model_name_or_path: str,
    samples: Sequence[Dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype | str,
    trust_remote_code: bool,
    enable_thinking: bool,
) -> List[str]:
    prompts = []
    for sample in samples:
        expert_text = str(sample.get("expert_text", "")).strip()
        if not expert_text:
            raise ValueError("text_distill baseline requested but sample is missing expert_text")
        prompts.append(build_text_distill_prompt(str(sample.get("question", "")), expert_text))
    outputs, _ = base.run_text_generation_stage(
        stage_name="long_context_text_distill",
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


@torch.no_grad()
def run_long_context_learner_stage(
    *,
    learner_model_name_or_path: str,
    questions: Sequence[str],
    source_latents: Sequence[torch.Tensor],
    outer_adapter_path: str,
    outer_adapter_type: str,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    model_dtype: torch.dtype | str,
    outer_dtype: torch.dtype | str,
    trust_remote_code: bool,
    enable_thinking: bool,
) -> Tuple[List[str], List[int], List[int]]:
    if not outer_adapter_path:
        raise ValueError("--outer_adapter_path is required for long_context_distillation mode")
    if len(questions) != len(source_latents):
        raise ValueError("questions/source_latents size mismatch")
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=learner_model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="long_context_learner",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype
    hidden_size = int(embed_layer.weight.size(-1))
    source_dim = int(source_latents[0].size(-1))
    outer = base.load_outer_adapter_module(
        adapter_path=outer_adapter_path,
        in_dim=source_dim,
        out_dim=hidden_size,
        adapter_type=outer_adapter_type,
        device=device,
        dtype=outer_dtype,
    )
    prompt_segments = [
        base.split_prompt_ids_by_slots(
            tokenizer,
            build_long_context_prompt_with_slot(question),
            [LONG_CONTEXT_LATENT_SLOT],
            enable_thinking,
        )
        for question in questions
    ]
    gen_kwargs = base.build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    outputs: List[str] = []
    latent_lengths: List[int] = []
    prompt_token_budgets: List[int] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(base.batch_iter_indices(len(questions), batch_size), total=total_batches, desc="long_context_distill"):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            prefix_ids, suffix_ids = prompt_segments[idx]
            prefix = base.token_ids_to_embeds(embed_layer, prefix_ids, device=device, dtype=embed_dtype)
            suffix = base.token_ids_to_embeds(embed_layer, suffix_ids, device=device, dtype=embed_dtype)
            mapped_latents = base.run_outer_adapter(
                outer,
                source_latents[idx].to(device=device, dtype=torch.float32),
                output_dtype=embed_dtype,
            )
            seq = torch.cat([prefix, mapped_latents, suffix], dim=0)
            embed_seqs.append(seq)
            latent_lengths.append(int(mapped_latents.size(0)))
            prompt_token_budgets.append(int(seq.size(0)))
        batch_embeds, attention_mask = base.pad_left_embeds(embed_seqs, device=device)
        generated = model.generate(inputs_embeds=batch_embeds, attention_mask=attention_mask, **gen_kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_len = attention_mask.size(1)
        gen_ids = sequences[:, prompt_len:] if sequences.size(1) > prompt_len else sequences
        outputs.extend([text.strip() for text in tokenizer.batch_decode(gen_ids, skip_special_tokens=True)])
    base.release_resources(model, tokenizer, outer)
    return outputs, latent_lengths, prompt_token_budgets


def maybe_write_training_placeholder(path: str, args: argparse.Namespace) -> None:
    if not path:
        return
    payload = {
        "style": RUN_MODE_LONG_CONTEXT,
        "prompt_slot": LONG_CONTEXT_LATENT_SLOT,
        "status": "placeholder_only",
        "notes": [
            "Training loop intentionally not implemented in the release scaffold.",
            "Teacher forcing should build inputs_embeds from prefix + latent slot + suffix + shifted answer embeddings.",
            "Labels should supervise the full answer autoregressively while masking prompt and latent positions.",
        ],
        "artifacts": {
            "source_latents": "precomputed .pt/.npy or teacher cache artifacts",
            "outer_adapter": args.outer_adapter_path,
            "compressor": args.compressor_path,
        },
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_write_results_jsonl(
    path: str,
    samples: Sequence[Dict[str, Any]],
    outputs_by_mode: Dict[str, Dict[str, Any]],
) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for mode_name, payload in outputs_by_mode.items():
            predictions = payload["predictions"]
            metrics = payload["metrics"]
            per_sample_metrics = {entry["sample_id"]: entry for entry in metrics.get("per_sample", [])}
            for sample_idx, (sample, prediction) in enumerate(zip(samples, predictions)):
                sample_id = sample.get("sample_id")
                row = {
                    "mode": mode_name,
                    "sample_id": sample_id,
                    "task": sample.get("task"),
                    "question": sample.get("question"),
                    "answer": sample.get("answer"),
                    "prediction": prediction,
                    "correct": per_sample_metrics.get(sample_id, {}).get("correct"),
                }
                if "latent_lengths" in payload:
                    row["latent_length"] = payload["latent_lengths"][sample_idx]
                if "prompt_token_budgets" in payload:
                    row["prompt_token_budget"] = payload["prompt_token_budgets"][sample_idx]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.method = RUN_MODE_LONG_CONTEXT
    base._GEN_TOP_K = int(args.top_k) if int(args.top_k) >= 0 else None
    base._GEN_MIN_P = float(args.min_p) if float(args.min_p) >= 0 else None
    base._GEN_REPETITION_PENALTY = float(args.repetition_penalty)
    if args.presence_penalty != 0.0:
        print("[warn] --presence_penalty is ignored by HF generation in this pipeline.")

    run_modes = parse_run_modes(args.run_modes)
    if any(mode in {RUN_MODE_LONG_CONTEXT, RUN_MODE_QUESTION_ONLY, RUN_MODE_TEXT_DISTILL} for mode in run_modes) and not args.learner_model_name_or_path:
        raise ValueError("--learner_model_name_or_path is required for generation modes")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")

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
    random.seed(args.synthetic_seed)
    torch.manual_seed(args.synthetic_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.synthetic_seed)

    samples = load_samples(args)
    maybe_write_training_placeholder(args.training_placeholder_path, args)
    questions = [str(sample.get("question", "")) for sample in samples]
    source_latents = build_source_latents(
        args,
        samples,
        device=device,
        model_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        enable_thinking=enable_thinking,
    )

    outputs_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in run_modes:
        if mode == RUN_MODE_LONG_CONTEXT:
            if source_latents is None:
                raise ValueError("long_context_distillation mode requires --long_context_latents_path or --teacher_model_name_or_path")
            predictions, latent_lengths, prompt_token_budgets = run_long_context_learner_stage(
                learner_model_name_or_path=args.learner_model_name_or_path,
                questions=questions,
                source_latents=source_latents,
                outer_adapter_path=args.outer_adapter_path,
                outer_adapter_type=args.outer_adapter_type_fallback,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                model_dtype=model_dtype,
                outer_dtype=outer_dtype,
                trust_remote_code=trust_remote_code,
                enable_thinking=enable_thinking,
            )
            metrics = compute_long_context_metrics(
                samples,
                predictions,
                latent_lengths=latent_lengths,
                prompt_token_budgets=prompt_token_budgets,
            )
            outputs_by_mode[mode] = {
                "predictions": predictions,
                "metrics": metrics,
                "latent_lengths": latent_lengths,
                "prompt_token_budgets": prompt_token_budgets,
            }
        elif mode == RUN_MODE_QUESTION_ONLY:
            predictions = run_question_only_baseline(
                learner_model_name_or_path=args.learner_model_name_or_path,
                questions=questions,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                dtype=model_dtype,
                trust_remote_code=trust_remote_code,
                enable_thinking=enable_thinking,
            )
            outputs_by_mode[mode] = {
                "predictions": predictions,
                "metrics": compute_long_context_metrics(samples, predictions),
            }
        elif mode == RUN_MODE_TEXT_DISTILL:
            predictions = run_text_distill_baseline(
                learner_model_name_or_path=args.learner_model_name_or_path,
                samples=samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                dtype=model_dtype,
                trust_remote_code=trust_remote_code,
                enable_thinking=enable_thinking,
            )
            outputs_by_mode[mode] = {
                "predictions": predictions,
                "metrics": compute_long_context_metrics(samples, predictions),
            }
        else:
            raise AssertionError(f"Unhandled mode: {mode}")

    maybe_write_results_jsonl(args.result_jsonl, samples, outputs_by_mode)
    for mode_name, payload in outputs_by_mode.items():
        metrics = dict(payload["metrics"])
        metrics.pop("per_sample", None)
        metric_text = ", ".join(f"{key}={value}" for key, value in metrics.items())
        print(f"[{mode_name}] {metric_text}")


if __name__ == "__main__":
    main()
