#!/usr/bin/env python
"""Context-equivalent checkpointed DKNet ablations on microblogPCU weak labels.

This runner applies the same previous-document LLM context rule as
`run_microblogpcu_dknet_context_checkpoint.py` while disabling individual
modules. It writes per-row predictions for each variant so interrupted runs can
be inspected and, with `--resume`, continued.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from run_microblogpcu_dknet_ablation_checkpoint import VARIANTS, apply_variant
from run_microblogpcu_dknet_checkpoint import (
    DEFAULT_INPUT,
    PROJECT_ROOT,
    configure_paths,
    load_config,
    load_weak_label_rows,
    make_document,
    make_jsonable,
    record_metrics,
)
from run_microblogpcu_dknet_context_checkpoint import read_existing_records, run_one_document
from run_microblogpcu_dknet_context_checkpoint import save_state, try_load_state


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "proxy_real_data" / "dknet_context_ablation_30"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "proxy_real_data" / "dknet_context_ablation_30_summary.json"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "proxy_real_data" / "dknet_context_ablation_30_summary.csv"
DEFAULT_STATE_DIR = PROJECT_ROOT / "proxy_real_data" / "dknet_context_ablation_30_state"


def replay_rows(model: Any, documents: list[dict[str, Any]], rows: list[dict[str, Any]], count: int) -> None:
    for index in range(count):
        run_one_document(model, documents, index, rows[index]["_labels"])


def run_variant(
    rows: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    variant: str,
    output_path: Path,
    state_path: Path,
    resume: bool,
    no_state_resume: bool,
    save_state_every: int,
) -> dict[str, Any]:
    from src.dknet_model import DKNet

    model = DKNet(load_config())
    variant_notes = apply_variant(model, variant)
    records = read_existing_records(output_path) if resume else []
    completed_count = len(records)
    resume_strategy = "fresh"
    if completed_count > len(rows):
        raise ValueError(f"{variant}: existing record count {completed_count} exceeds requested rows {len(rows)}")
    if completed_count:
        state_loaded = False if no_state_resume else try_load_state(model, state_path, completed_count)
        if state_loaded:
            resume_strategy = "state_snapshot"
        else:
            replay_rows(model, documents, rows, completed_count)
            save_state(model, state_path, completed_count)
            resume_strategy = "exact_replay"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and completed_count else "w"
    started_at = time.time()
    with output_path.open(mode, encoding="utf-8") as out_fh:
        for index, row in enumerate(rows[completed_count:], start=completed_count):
            row_started = time.time()
            record: dict[str, Any] = {
                "row_index": index + 1,
                "variant": variant,
                "source_post_id": row.get("source_post_id", ""),
                "labels": row["_labels"],
                "status": "ok",
                "error": None,
                "elapsed_seconds": 0.0,
                "context_size": index,
                "predicted": [],
                "stats": {},
            }
            try:
                predictions, stats = run_one_document(model, documents, index, row["_labels"])
                record["predicted"] = make_jsonable(predictions)
                record["stats"] = make_jsonable(stats)
                record["context_size"] = stats["context_size"]
            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
            record["elapsed_seconds"] = time.time() - row_started
            records.append(record)
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            if save_state_every > 0 and len(records) % save_state_every == 0:
                save_state(model, state_path, len(records))

    save_state(model, state_path, len(records))

    ok_records = [record for record in records if record["status"] == "ok"]
    error_records = [record for record in records if record["status"] != "ok"]
    empty_prediction_records = [record for record in ok_records if not record["predicted"]]
    metrics = [record_metrics(record) for record in ok_records]
    row_times = [record["elapsed_seconds"] for record in ok_records]

    def mean(key: str) -> float:
        return sum(metric[key] for metric in metrics) / len(metrics) if metrics else 0.0

    return {
        "variant": variant,
        "output": str(output_path),
        "requested_rows": len(rows),
        "completed_rows": len(ok_records),
        "error_rows": len(error_records),
        "empty_prediction_rows": len(empty_prediction_records),
        "elapsed_seconds": time.time() - started_at,
        "avg_row_seconds": sum(row_times) / len(row_times) if row_times else 0.0,
        "max_context_size": max((record.get("context_size", 0) for record in ok_records), default=0),
        "resume_strategy": resume_strategy,
        "precision_at_10": mean("precision_at_10"),
        "recall_at_10": mean("recall_at_10"),
        "f1_at_10": mean("f1_at_10"),
        "hit_at_10": mean("hit_at_10"),
        "notes": variant_notes,
        "error_examples": [record["error"] for record in error_records[:3]],
    }


def add_deltas(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = next(item for item in summaries if item["variant"] == "full")
    for item in summaries:
        for metric in ("precision_at_10", "recall_at_10", "f1_at_10", "hit_at_10"):
            item[f"delta_{metric}_vs_full"] = item[metric] - full[metric]
    return summaries


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "requested_rows",
        "completed_rows",
        "error_rows",
        "empty_prediction_rows",
        "elapsed_seconds",
        "avg_row_seconds",
        "max_context_size",
        "resume_strategy",
        "precision_at_10",
        "recall_at_10",
        "f1_at_10",
        "hit_at_10",
        "delta_precision_at_10_vs_full",
        "delta_recall_at_10_vs_full",
        "delta_f1_at_10_vs_full",
        "delta_hit_at_10_vs_full",
        "output",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow({field: item.get(field, "") for field in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-state-every", type=int, default=1)
    parser.add_argument("--no-state-resume", action="store_true")
    args = parser.parse_args()

    configure_paths()
    rows = load_weak_label_rows(args.input, args.limit)
    documents = [make_document(row) for row in rows]
    summaries = []
    for variant in args.variants:
        output_path = args.output_dir / f"{variant}.jsonl"
        state_path = args.state_dir / f"{variant}.pkl"
        summaries.append(
            run_variant(
                rows,
                documents,
                variant,
                output_path,
                state_path,
                args.resume,
                args.no_state_resume,
                args.save_state_every,
            )
        )

    summaries = add_deltas(summaries)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "input": str(args.input),
                "limit": args.limit,
                "variants": list(args.variants),
                "notes": [
                    "Context-equivalent checkpoint ablation run.",
                    "Each row receives all previous row texts as LLM context.",
                    "Predictions are written after each row for auditability and resumability.",
                    "When --resume is used, completed rows are restored either from a matching per-variant state snapshot or by exact replay.",
                    "Weak hashtags are proxy labels and not expert annotations.",
                ],
                "summaries": summaries,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    write_summary_csv(args.summary_csv, summaries)
    print(json.dumps({"summary_json": str(args.summary_json), "summary_csv": str(args.summary_csv), "summaries": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
