"""Render the side-by-side protocol comparison as Markdown.

Usage: python scripts/write_results.py --output results/PROTOCOL.md NAME=metrics.json [NAME=metrics.json ...]

Each metrics file is either a ``protocol_metrics.json`` written by ``scripts/score_protocol.py`` or the
``metrics.json`` written by ``ubs-forecast ensemble`` (its ``ensemble`` and ``components`` entries are
expanded into rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ubs_forecasting.vocab import CLASSES


def load_rows(specs: list[str]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for spec in specs:
        name, _, path = spec.partition("=")
        payload = json.loads(Path(path).read_text())
        if "ensemble" in payload:
            for component, result in payload["components"].items():
                rows.append((f"{name}: {component}", result))
            rows.append((name, payload["ensemble"]))
        else:
            rows.append((name, payload))
    return rows


def interval(values: list[float]) -> str:
    return f"{values[0]:.3f}-{values[1]:.3f}"


def render(rows: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "# Shared-protocol comparison",
        "",
        "Development set: all train clients plus non-lockbox valid clients (2,791). Folds: five hashed client folds "
        "repeated for seeds 0, 1, 2. Lockbox: the 209 valid clients whose hashed identifier is divisible by five, "
        "predicted once by the final model. Macro-F1 is over the fixed eight labels. Intervals are 95% client "
        "bootstraps (2,000 resamples; the same resample is applied across seeds).",
        "",
        "Validation-only and lockbox are the headline numbers: pooled scores mix in the much cleaner train clients. "
        "Differences under about 0.02 are within noise.",
        "",
        "| Solution | Pooled CV (95% CI) | Validation-only CV (95% CI) | Lockbox (95% CI) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in rows:
        development = result.get("development")
        lockbox = result.get("lockbox")
        pooled = (
            f"{development['pooled_macro_f1']:.3f} ({interval(development['pooled_ci95'])})" if development else "-"
        )
        valid = (
            f"{development['valid_only_macro_f1']:.3f} ({interval(development['valid_only_ci95'])})"
            if development
            else "-"
        )
        locked = f"{lockbox['macro_f1']:.3f} ({interval(lockbox['ci95'])})" if lockbox else "-"
        lines.append(f"| {name} | {pooled} | {valid} | {locked} |")
    lines.extend(
        [
            "",
            "## Per-seed validation-only macro-F1",
            "",
            "| Solution | Seed 0 | Seed 1 | Seed 2 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, result in rows:
        development = result.get("development")
        if not development:
            continue
        seeds = development["seeds"]
        lines.append(
            f"| {name} | " + " | ".join(f"{seeds[str(seed)]['valid_only_macro_f1']:.3f}" for seed in (0, 1, 2)) + " |"
        )
    lines.extend(
        [
            "",
            "## Per-class F1 (validation-only, mean over seeds)",
            "",
            "| Solution | " + " | ".join(CLASSES) + " |",
            "| --- | " + " | ".join("---:" for _ in CLASSES) + " |",
        ]
    )
    for name, result in rows:
        development = result.get("development")
        if not development:
            continue
        per_class = development["valid_only_per_class_f1"]
        lines.append(f"| {name} | " + " | ".join(f"{per_class[label]:.3f}" for label in CLASSES) + " |")
    lines.extend(
        [
            "",
            "## Per-class F1 (lockbox)",
            "",
            "| Solution | " + " | ".join(CLASSES) + " |",
            "| --- | " + " | ".join("---:" for _ in CLASSES) + " |",
        ]
    )
    for name, result in rows:
        lockbox = result.get("lockbox")
        if not lockbox:
            continue
        per_class = lockbox["per_class_f1"]
        lines.append(f"| {name} | " + " | ".join(f"{per_class[label]:.3f}" for label in CLASSES) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("specs", nargs="+", help="NAME=path/to/metrics.json")
    args = parser.parse_args()
    text = render(load_rows(args.specs))
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
