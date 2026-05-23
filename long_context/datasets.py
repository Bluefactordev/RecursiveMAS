from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


TaskSample = Dict[str, Any]


def _filler_lines(rng: random.Random, total_lines: int) -> List[str]:
    topics = [
        "warehouse inventory",
        "weather archive",
        "meeting notes",
        "shipping manifest",
        "lab notebook",
        "library index",
        "project changelog",
        "backup ledger",
    ]
    return [
        f"Background note {idx:02d}: {rng.choice(topics)} entry token={rng.randint(1000, 9999)}."
        for idx in range(total_lines)
    ]


def _build_needle_sample(rng: random.Random, sample_id: int, filler_lines: int) -> TaskSample:
    answer = f"NEEDLE-{sample_id:03d}-{rng.randint(100, 999)}"
    lines = _filler_lines(rng, filler_lines)
    insert_at = rng.randint(max(1, filler_lines // 4), max(1, filler_lines - 1))
    lines.insert(insert_at, f"Critical record: the verification code is {answer}.")
    question = "What is the verification code mentioned in the context?"
    return {
        "sample_id": f"needle-{sample_id:03d}",
        "task": "needle",
        "context": "\n".join(lines),
        "question": question,
        "answer": answer,
        "expert_text": "Step 1: Locate the single critical record.\nStep 2: Copy the verification code exactly.",
        "metadata": {"filler_lines": filler_lines, "insert_at": insert_at},
    }


def _build_latest_value_sample(rng: random.Random, sample_id: int, filler_lines: int) -> TaskSample:
    key = f"service-{sample_id:03d}"
    latest_value = f"v{rng.randint(20, 40)}.{rng.randint(0, 9)}"
    version_lines = [
        f"Configuration update {idx + 1}: {key} = v{idx + 1}.{rng.randint(0, 9)}"
        for idx in range(3)
    ]
    version_lines.append(f"Configuration update 4: {key} = {latest_value}")
    lines = _filler_lines(rng, filler_lines)
    lines.extend(version_lines)
    rng.shuffle(lines)
    question = f"What is the latest configured value for {key}?"
    return {
        "sample_id": f"latest-{sample_id:03d}",
        "task": "latest_value",
        "context": "\n".join(lines),
        "question": question,
        "answer": latest_value,
        "expert_text": "Step 1: Find every configuration update for the target service.\nStep 2: Return the newest value only.",
        "metadata": {"service": key},
    }


def _build_two_hop_sample(rng: random.Random, sample_id: int, filler_lines: int) -> TaskSample:
    researcher = f"Researcher-{sample_id:03d}"
    project = f"Project-{rng.randint(100, 999)}"
    artifact = f"ART-{rng.randint(1000, 9999)}"
    lines = _filler_lines(rng, filler_lines)
    lines.extend(
        [
            f"Assignment note: {researcher} maintains {project}.",
            f"Archive note: {project} stores artifact {artifact}.",
        ]
    )
    rng.shuffle(lines)
    question = f"Which artifact is associated with {researcher}?"
    return {
        "sample_id": f"twohop-{sample_id:03d}",
        "task": "two_hop",
        "context": "\n".join(lines),
        "question": question,
        "answer": artifact,
        "expert_text": "Step 1: Map the researcher to the project.\nStep 2: Map that project to its artifact and copy it exactly.",
        "metadata": {"researcher": researcher, "project": project},
    }


def _build_symbolic_copy_sample(rng: random.Random, sample_id: int, filler_lines: int) -> TaskSample:
    fields = {
        "code": f"ZX-{rng.randint(1000, 9999)}-{rng.choice(['AA', 'BB', 'CC'])}",
        "date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "id": f"ID{rng.randint(100000, 999999)}",
    }
    target_field = rng.choice(list(fields.keys()))
    lines = _filler_lines(rng, filler_lines)
    lines.extend([f"Symbolic record: {name} => {value}" for name, value in fields.items()])
    rng.shuffle(lines)
    question = f"Copy the {target_field} value exactly from the symbolic record."
    return {
        "sample_id": f"symbolic-{sample_id:03d}",
        "task": "symbolic_copy",
        "context": "\n".join(lines),
        "question": question,
        "answer": fields[target_field],
        "expert_text": "Step 1: Find the symbolic record for the requested field.\nStep 2: Copy the value exactly without paraphrasing.",
        "metadata": {"target_field": target_field, "all_fields": fields},
    }


_SYNTHETIC_BUILDERS = {
    "needle": _build_needle_sample,
    "latest_value": _build_latest_value_sample,
    "two_hop": _build_two_hop_sample,
    "symbolic_copy": _build_symbolic_copy_sample,
}


def build_synthetic_long_context_dataset(
    num_samples: int,
    *,
    seed: int = 42,
    tasks: Optional[Sequence[str]] = None,
    filler_lines: int = 24,
) -> List[TaskSample]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    task_names = [str(task).strip().lower() for task in (tasks or _SYNTHETIC_BUILDERS.keys())]
    for task_name in task_names:
        if task_name not in _SYNTHETIC_BUILDERS:
            raise ValueError(f"Unsupported synthetic task: {task_name}")
    rng = random.Random(seed)
    samples: List[TaskSample] = []
    for idx in range(num_samples):
        builder = _SYNTHETIC_BUILDERS[task_names[idx % len(task_names)]]
        samples.append(builder(rng, idx, filler_lines))
    return samples


def _normalize_samples(samples: Iterable[Dict[str, Any]]) -> List[TaskSample]:
    out: List[TaskSample] = []
    for idx, sample in enumerate(samples):
        normalized = dict(sample)
        normalized.setdefault("sample_id", f"sample-{idx:05d}")
        normalized.setdefault("task", "custom")
        normalized.setdefault("context", "")
        normalized.setdefault("question", "")
        normalized.setdefault("answer", "")
        normalized.setdefault("metadata", {})
        out.append(normalized)
    return out


def load_long_context_samples(path: str | Path) -> List[TaskSample]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            return _normalize_samples(
                json.loads(line)
                for line in handle
                if line.strip()
            )
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        samples = payload.get("samples", [])
    else:
        samples = payload
    if not isinstance(samples, list):
        raise ValueError("Dataset payload must be a list or {\"samples\": [...]}.")
    return _normalize_samples(samples)


def save_long_context_samples(path: str | Path, samples: Sequence[TaskSample]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_samples(samples)
    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for sample in normalized:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2)
