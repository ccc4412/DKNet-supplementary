#!/usr/bin/env python
"""Checkpointed DKNet ablation feasibility runs on microblogPCU weak labels."""

from __future__ import annotations

import argparse
import csv
import json
import types
from pathlib import Path
from typing import Any

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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "proxy_real_data" / "dknet_checkpoint_ablation_100"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "proxy_real_data" / "dknet_checkpoint_ablation_100_summary.json"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "proxy_real_data" / "dknet_checkpoint_ablation_100_summary.csv"
VARIANTS = ("full", "no_kg", "no_mmr", "no_adaptive")


def apply_variant(model: Any, variant: str) -> list[str]:
    notes: list[str] = []
    if variant == "full":
        notes.append("Original DKNet components retained.")
        return notes

    if variant == "no_kg":
        model._get_kg_candidates = types.MethodType(lambda self, text, timestamp=None: {}, model)
        notes.append("KG candidate contribution disabled by returning an empty KG candidate dictionary.")
        return notes

    if variant == "no_mmr":
        def no_mmr(self: Any, candidates: list[tuple[str, float]], embeddings: dict[str, Any], query_embedding: Any = None, top_k: int = 10, lambda_diversity: float | None = None) -> list[tuple[str, float]]:
            return sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k]

        model.mmr_optimizer.mmr_diversify = types.MethodType(no_mmr, model.mmr_optimizer)
        notes.append("MMR diversification disabled by selecting top fused scores directly.")
        return notes

    if variant == "no_adaptive":
        def fixed_fusion(self: Any, kg_scores: Any, llm_scores: Any, update_weights: bool = True) -> tuple[Any, float]:
            alpha = self.config["model"]["adaptive"]["alpha_init"]
            fusion_scores = alpha * kg_scores + (1 - alpha) * llm_scores
            self.fusion_scores_history.append(fusion_scores.copy())
            return fusion_scores, alpha

        model.adaptive_fusion.batch_adaptive_fusion = types.MethodType(fixed_fusion, model.adaptive_fusion)
        notes.append("Adaptive weight updates disabled; alpha fixed at alpha_init.")
        return notes

    raise ValueError(f"Unknown variant: {variant}")


def run_variant(rows: list[dict[str, Any]], variant: str, output_path: Path) -> dict[str, Any]:
    from src.dknet_model import DKNet

    model = DKNet(load_config())
    variant_notes = apply_variant(model, variant)
    records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_fh:
        for idx, row in enumerate(rows, start=1):
            record: dict[str, Any] = {
                "row_index": idx,
                "variant": variant,
                "source_post_id": row.get("source_post_id", ""),
                "labels": row["_labels"],
                "status": "ok",
                "error": None,
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
            records.append(record)
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()

    ok_records = [record for record in records if record["status"] == "ok"]
    error_records = [record for record in records if record["status"] != "ok"]
    empty_prediction_records = [record for record in ok_records if not record["predicted"]]
    metrics = [record_metrics(record) for record in ok_records]

    def mean(key: str) -> float:
        return sum(metric[key] for metric in metrics) / len(metrics) if metrics else 0.0

    return {
        "variant": variant,
        "output": str(output_path),
        "requested_rows": len(rows),
        "completed_rows": len(ok_records),
        "error_rows": len(error_records),
        "empty_prediction_rows": len(empty_prediction_records),
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
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    configure_paths()
    rows = load_weak_label_rows(args.input, args.limit)
    summaries = []
    for variant in args.variants:
        output_path = args.output_dir / f"{variant}.jsonl"
        summaries.append(run_variant(rows, variant, output_path))

    summaries = add_deltas(summaries)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "input": str(args.input),
                "limit": args.limit,
                "variants": list(args.variants),
                "notes": [
                    "Checkpoint ablation feasibility run; rows are submitted one at a time.",
                    "This is not one-shot-context equivalent to DKNet.run(documents).",
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
