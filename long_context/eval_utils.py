from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


Metrics = Dict[str, Any]


def normalize_eval_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


def compute_long_context_metrics(
    samples: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    *,
    latent_lengths: Optional[Sequence[int]] = None,
    prompt_token_budgets: Optional[Sequence[int]] = None,
) -> Metrics:
    if len(samples) != len(predictions):
        raise ValueError("samples/predictions size mismatch")
    task_totals: Dict[str, int] = {}
    task_correct: Dict[str, int] = {}
    exact_correct = 0
    per_sample = []
    for idx, sample in enumerate(samples):
        task_name = str(sample.get("task", "custom"))
        gold = normalize_eval_text(str(sample.get("answer", "")))
        pred = normalize_eval_text(predictions[idx])
        is_correct = int(pred == gold)
        exact_correct += is_correct
        task_totals[task_name] = task_totals.get(task_name, 0) + 1
        task_correct[task_name] = task_correct.get(task_name, 0) + is_correct
        per_sample.append(
            {
                "sample_id": sample.get("sample_id", idx),
                "task": task_name,
                "prediction": predictions[idx],
                "gold": sample.get("answer", ""),
                "correct": bool(is_correct),
            }
        )

    metrics: Metrics = {
        "num_samples": len(samples),
        "answer_exact_match": _accuracy(exact_correct, len(samples)),
        "symbolic_copy_accuracy": _accuracy(task_correct.get("symbolic_copy", 0), task_totals.get("symbolic_copy", 0)),
        "latest_value_accuracy": _accuracy(task_correct.get("latest_value", 0), task_totals.get("latest_value", 0)),
        "multi_hop_accuracy": _accuracy(task_correct.get("two_hop", 0), task_totals.get("two_hop", 0)),
        "needle_accuracy": _accuracy(task_correct.get("needle", 0), task_totals.get("needle", 0)),
        "per_sample": per_sample,
    }
    if latent_lengths is not None:
        lengths = [int(length) for length in latent_lengths]
        metrics["latent_length_mean"] = (sum(lengths) / len(lengths)) if lengths else 0.0
        metrics["latent_length_max"] = max(lengths) if lengths else 0
    if prompt_token_budgets is not None:
        budgets = [int(length) for length in prompt_token_budgets]
        metrics["prompt_token_budget_mean"] = (sum(budgets) / len(budgets)) if budgets else 0.0
        metrics["prompt_token_budget_max"] = max(budgets) if budgets else 0
    return metrics
