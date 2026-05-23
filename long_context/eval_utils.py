from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def evaluate_long_context_predictions(
    samples: Sequence[Dict[str, str]],
    predictions: Sequence[str],
    latent_lengths: Optional[Sequence[int]] = None,
    prompt_token_lengths: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    total = max(1, len(samples))
    exact = 0
    symbolic = 0
    latest = 0
    two_hop = 0
    symbolic_total = 0
    latest_total = 0
    two_hop_total = 0

    for i, sample in enumerate(samples):
        pred = predictions[i] if i < len(predictions) else ""
        gold = sample.get("answer", "")
        ok = _norm(pred) == _norm(gold)
        if ok:
            exact += 1

        task = sample.get("task_type", "")
        if task == "symbolic_copy":
            symbolic_total += 1
            if str(pred).strip() == str(gold).strip():
                symbolic += 1
        elif task == "latest_value":
            latest_total += 1
            if ok:
                latest += 1
        elif task == "two_hop":
            two_hop_total += 1
            if ok:
                two_hop += 1

    metrics: Dict[str, float] = {
        "answer_exact_match": 100.0 * exact / total,
        "symbolic_copy_accuracy": 100.0 * symbolic / max(1, symbolic_total),
        "latest_value_accuracy": 100.0 * latest / max(1, latest_total),
        "multi_hop_accuracy": 100.0 * two_hop / max(1, two_hop_total),
        "num_samples": float(len(samples)),
    }

    if latent_lengths is not None and len(latent_lengths) > 0:
        ll = [int(x) for x in latent_lengths]
        metrics["avg_latent_length"] = float(sum(ll)) / len(ll)
        metrics["max_latent_length"] = float(max(ll))
    if prompt_token_lengths is not None and len(prompt_token_lengths) > 0:
        pl = [int(x) for x in prompt_token_lengths]
        metrics["avg_prompt_token_budget"] = float(sum(pl)) / len(pl)
        metrics["max_prompt_token_budget"] = float(max(pl))
    return metrics
