#!/usr/bin/env python
"""Run a small auditable DKNet smoke test on the microblogPCU proxy data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION2_ROOT = PROJECT_ROOT.parent / "version2"
PYTHON_DEPS = PROJECT_ROOT / "python_deps"
DEFAULT_INPUT = PROJECT_ROOT / "proxy_real_data" / "microblogpcu_dknet_proxy.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "proxy_real_data" / "microblogpcu_dknet_smoke.json"


def configure_paths() -> None:
    for path in (PYTHON_DEPS, VERSION2_ROOT):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_config() -> dict[str, Any]:
    config_path = VERSION2_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    config["model"]["llm"]["device"] = "cpu"
    config["model"]["llm"]["batch_size"] = 1
    config.setdefault("data", {}).setdefault("preprocessing", {})
    config["data"]["preprocessing"]["min_text_length"] = 1
    config["data"]["preprocessing"]["max_text_length"] = 500
    config.setdefault("experiment", {})["top_k"] = 10
    return config


def parse_keywords(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]


def load_rows(path: Path, limit: int) -> tuple[list[dict[str, Any]], list[list[str]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    labels: list[list[str]] = []
    raw_rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            kws = parse_keywords(row.get("keywords", ""))
            if not kws:
                continue
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
            except Exception:
                timestamp = datetime.now()
            documents.append(
                {
                    "text": row["text"],
                    "timestamp": timestamp,
                    "source_post_id": row.get("source_post_id", ""),
                }
            )
            labels.append(kws)
            raw_rows.append(row)
            if len(documents) >= limit:
                break

    return documents, labels, raw_rows


def normalize_prediction(item: Any) -> str:
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    return str(item)


def evaluate(predictions: list[list[Any]], labels: list[list[str]], k: int = 10) -> dict[str, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    hits: list[float] = []

    for pred, truth in zip(predictions, labels):
        pred_terms = [normalize_prediction(item) for item in pred[:k]]
        pred_set = set(pred_terms)
        truth_set = set(truth)
        overlap = len(pred_set & truth_set)
        precisions.append(overlap / k)
        recalls.append(overlap / len(truth_set) if truth_set else 0.0)
        hits.append(1.0 if overlap else 0.0)

    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hit = sum(hits) / len(hits) if hits else 0.0
    return {"precision_at_10": precision, "recall_at_10": recall, "f1_at_10": f1, "hit_at_10": hit}


def make_jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    configure_paths()
    from src.dknet_model import DKNet

    documents, labels, raw_rows = load_rows(args.input, args.limit)
    started = time.time()
    status = "ok"
    error: str | None = None
    predictions: list[list[Any]] = []
    stats: dict[str, Any] = {}

    try:
        model = DKNet(load_config())
        predictions, stats = model.run(documents, ground_truth_keywords=labels)
    except Exception as exc:
        status = "error"
        error = repr(exc)

    elapsed = time.time() - started
    output = {
        "status": status,
        "error": error,
        "input": str(args.input),
        "sample_size": len(documents),
        "weak_label_rows": len(labels),
        "elapsed_seconds": elapsed,
        "metrics": evaluate(predictions, labels) if predictions else {},
        "predictions": [
            {
                "source_post_id": row.get("source_post_id", ""),
                "labels": label,
                "predicted": make_jsonable(pred),
            }
            for row, label, pred in zip(raw_rows, labels, predictions)
        ],
        "stats": make_jsonable(stats),
        "notes": [
            "Smoke test only; sample is not a manuscript-scale validation.",
            "Weak hashtags are used as proxy labels and are not expert annotations.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"status": status, "output": str(args.output), "elapsed_seconds": elapsed}, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
