from __future__ import annotations

import random
from typing import Dict, List


def _rand_token(rng: random.Random, n: int = 8) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(n))


def _filler_lines(rng: random.Random, n: int) -> List[str]:
    return [f"Noise line {i}: {_rand_token(rng, 12)}" for i in range(n)]


def build_synthetic_long_context_samples(num_samples: int = 32, seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    tasks = ["needle", "latest_value", "two_hop", "symbolic_copy"]
    samples: List[Dict[str, str]] = []

    for i in range(max(1, num_samples)):
        task = tasks[i % len(tasks)]
        if task == "needle":
            key = f"needle_key_{i}"
            value = _rand_token(rng, 10)
            lines = _filler_lines(rng, 40)
            pos = rng.randint(5, 35)
            lines.insert(pos, f"Record: {key} = {value}")
            context = "\n".join(lines)
            question = f"What is the exact value of {key}? Return only the value."
            answer = value
        elif task == "latest_value":
            key = f"setting_{i}"
            values = [_rand_token(rng, 6) for _ in range(3)]
            lines = _filler_lines(rng, 30)
            lines += [
                f"v1 update: {key}={values[0]}",
                f"v2 update: {key}={values[1]}",
                f"v3 update: {key}={values[2]}",
            ]
            rng.shuffle(lines)
            lines += [f"Latest authoritative update: {key}={values[2]}"]
            context = "\n".join(lines)
            question = f"What is the latest value for {key}? Return only the final value."
            answer = values[2]
        elif task == "two_hop":
            city = f"City-{_rand_token(rng, 4)}"
            country = f"Country-{_rand_token(rng, 4)}"
            currency = f"CUR-{_rand_token(rng, 3)}"
            lines = _filler_lines(rng, 35)
            lines += [
                f"Fact A: {city} belongs to {country}.",
                f"Fact B: The official currency of {country} is {currency}.",
            ]
            context = "\n".join(lines)
            question = f"What is the official currency used in {city}? Return only the currency code."
            answer = currency
        else:
            rec_id = f"ID-{_rand_token(rng, 5)}"
            date = f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
            code = f"CODE-{_rand_token(rng, 7)}"
            lines = _filler_lines(rng, 25)
            lines += [f"Symbolic record => REC_ID:{rec_id} DATE:{date} ACCESS_CODE:{code}"]
            context = "\n".join(lines)
            question = "Copy the ACCESS_CODE exactly from the record."
            answer = code

        samples.append(
            {
                "id": str(i),
                "task_type": task,
                "context": context,
                "question": question,
                "answer": answer,
            }
        )
    return samples
