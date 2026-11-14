#!/usr/bin/env python
"""Checkpointed DKNet feasibility run on microblogPCU weak-label rows.

This script keeps one DKNet instance alive and writes one prediction record at a
time. It is designed for long-running feasibility audits where a one-shot
`DKNet.run(documents)` call may time out before producing an artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_OUTPUT = PROJECT_ROOT / "proxy_real_data" / "microblogpcu_dknet_checkpoint_100.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "proxy_real_data" / "microblogpcu_dknet_checkpoint_100_summary.json"


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


def load_weak_label_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels = parse_keywords(row.get("keywords", ""))
            if not labels:
                continue
            row["_labels"] = labels
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def normalize_prediction(item: Any) -> str:
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    return str(item)


def make_document(row: dict[str, Any]) -> dict[str, Any]:
    try:
        timestamp = datetime.fromisoformat(row["timestamp"])
    except Exception:
        timestamp = datetime.now()
    return {
        "text": row["text"],
        "timestamp": timestamp,
        "source_post_id": row.get("source_post_id", ""),
    }


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


def record_metrics(record: dict[str, Any], k: int = 10) -> dict[str, float]:
    labels = set(record["labels"])
    predicted = [normalize_prediction(item) for item in record["predicted"][:k]]
    overlap = len(set(predicted) & labels)
    precision = overlap / k
    recall = overlap / len(labels) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "f1_at_10": f1,
        "hit_at_10": 1.0 if overlap else 0.0,
    }


def summarize(records: list[dict[str, Any]], started_at: float, input_path: Path, output_path: Path) -> dict[str, Any]:
    ok_records = [r for r in records if r.get("status") == "ok"]
    metrics = [record_metrics(r) for r in ok_records]

    def mean(key: str) -> float:
        return sum(m[key] for m in metrics) / len(metrics) if metrics else 0.0

    times = [r.get("elapsed_seconds", 0.0) for r in ok_records]
    summary = {
        "status": "ok" if len(ok_records) == len(records) else "partial",
        "input": str(input_path),
        "output": str(output_path),
        "requested_rows": len(records),
        "completed_rows": len(ok_records),
        "elapsed_seconds": time.time() - started_at,
        "avg_row_seconds": sum(times) / len(times) if times else 0.0,
        "metrics": {
            "precision_at_10": mean("precision_at_10"),
            "recall_at_10": mean("recall_at_10"),
            "f1_at_10": mean("f1_at_10"),
            "hit_at_10": mean("hit_at_10"),
        },
        "notes": [
            "Checkpoint feasibility run; each row is written immediately.",
            "The same DKNet instance is retained, so temporal KG and adaptive weights persist across rows.",
            "Rows are submitted one at a time; LLM context_texts are therefore not equivalent to a one-shot DKNet.run(documents) call.",
            "When --resume is used, completed rows are replayed first to restore temporal KG and adaptive-weight state.",
            "Weak hashtags are proxy labels and not expert annotations.",
        ],
    }
    return summary


def read_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def replay_rows(model: Any, rows: list[dict[str, Any]], count: int) -> None:
    for row in rows[:count]:
        model.run([make_document(row)], ground_truth_keywords=[row["_labels"]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    configure_paths()
    from src.dknet_model import DKNet

    rows = load_weak_label_rows(args.input, args.limit)
    model = DKNet(load_config())
    records = read_existing_records(args.output) if args.resume else []
    completed_count = len(records)
    if completed_count > len(rows):
        raise ValueError(f"Existing record count {completed_count} exceeds requested rows {len(rows)}")
    if completed_count:
        replay_rows(model, rows, completed_count)
    started_at = time.time()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and completed_count else "w"
    with args.output.open(mode, encoding="utf-8") as out_fh:
        for idx, row in enumerate(rows[completed_count:], start=completed_count + 1):
            row_started = time.time()
            record: dict[str, Any] = {
                "row_index": idx,
                "source_post_id": row.get("source_post_id", ""),
                "labels": row["_labels"],
                "status": "ok",
                "error": None,
                "elapsed_seconds": 0.0,
                "predicted": [],
                "stats": {},
            }
            try:
                predictions, stats = model.run([make_document(row)], ground_truth_keywords=[row["_labels"]])
                record["predicted"] = make_jsonable(predictions[0] if predictions else [])
                record["stats"] = make_jsonable(stats)
            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
            record["elapsed_seconds"] = time.time() - row_started
            records.append(record)
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()

    summary = summarize(records, started_at, args.input, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
