#!/usr/bin/env python
"""Context-equivalent checkpointed DKNet run on microblogPCU weak-label rows.

Unlike the simpler checkpoint runner, this script preserves the one-shot
`DKNet.run(documents)` context semantics by passing all previous document texts
to the LLM candidate generator for each row while still writing each prediction
immediately.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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


DEFAULT_OUTPUT = PROJECT_ROOT / "proxy_real_data" / "microblogpcu_dknet_context_checkpoint_30.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "proxy_real_data" / "microblogpcu_dknet_context_checkpoint_30_summary.json"
DEFAULT_STATE = PROJECT_ROOT / "proxy_real_data" / "microblogpcu_dknet_context_checkpoint_30_state.pkl"


def read_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_one_document(model: Any, documents: list[dict[str, Any]], index: int, labels: list[str]) -> tuple[list[Any], dict[str, Any]]:
    doc = documents[index]
    started = time.time()
    processed_text = model.data_processor.preprocess_text(doc["text"])

    model.temporal_kg.build_from_documents(
        [doc],
        model.data_processor,
        doc.get("timestamp", datetime.now()),
    )

    llm_candidates = model.llm_generator.generate_candidates_with_context(
        processed_text,
        context_texts=[previous["text"] for previous in documents[:index]],
        top_k=model.config.get("model", {}).get("llm", {}).get("max_candidates", 20),
    )
    kg_candidates = model._get_kg_candidates(processed_text, doc.get("timestamp"))

    kg_keys = list(kg_candidates.keys())
    llm_keys = [word for word, _, _ in llm_candidates]
    all_candidates = list(set(kg_keys + llm_keys))

    aligned_kg_scores = []
    aligned_llm_scores = []
    for candidate in all_candidates:
        kg_score = kg_candidates.get(candidate, 0.0)
        llm_score = next((score for word, score, _ in llm_candidates if word == candidate), 0.0)
        aligned_kg_scores.append(kg_score)
        aligned_llm_scores.append(llm_score)

    kg_scores_array = np.array(aligned_kg_scores)
    llm_scores_array = np.array(aligned_llm_scores)
    fusion_scores_array, _ = model.adaptive_fusion.batch_adaptive_fusion(
        kg_scores_array,
        llm_scores_array,
        labels,
    )
    fusion_scores = {
        candidate: score
        for candidate, score in zip(all_candidates, fusion_scores_array)
    }
    candidates = [(word, score) for word, score in fusion_scores.items()]

    embeddings = {}
    for word, _ in candidates:
        embedding = model.data_processor.get_text_embedding(word)
        if embedding is not None:
            embeddings[word] = embedding

    query_embedding = model.data_processor.get_text_embedding(processed_text)
    final_keywords = model.mmr_optimizer.mmr_diversify(
        candidates,
        embeddings,
        query_embedding,
        top_k=model.config.get("experiment", {}).get("top_k", 10),
    )

    stats = {
        "processing_time": time.time() - started,
        "context_size": index,
        "kg_stats": make_jsonable(model.temporal_kg.get_statistics()),
        "fusion_weights": make_jsonable(model.adaptive_fusion.get_weight_statistics()),
        "candidate_count": len(candidates),
    }
    return final_keywords, stats


def replay_rows(
    model: Any,
    documents: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    count: int,
    start: int = 0,
) -> None:
    for index in range(start, count):
        run_one_document(model, documents, index, rows[index]["_labels"])


def save_state(model: Any, path: Path, completed_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_count": completed_count,
        "temporal_kg": model.temporal_kg,
        "adaptive_fusion": model.adaptive_fusion,
        "notes": [
            "State snapshot for context-equivalent microblogPCU checkpoint resume.",
            "Only temporal KG and adaptive-fusion state are stored; model encoders are reloaded normally.",
        ],
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def try_load_state(model: Any, path: Path, completed_count: int) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if payload.get("completed_count") != completed_count:
        return False
    model.temporal_kg = payload["temporal_kg"]
    model.adaptive_fusion = payload["adaptive_fusion"]
    return True


def try_load_state_prefix(model: Any, path: Path, completed_count: int) -> int | None:
    """Load a snapshot that is aligned with a prefix of the JSONL checkpoint."""
    if not path.exists():
        return None
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    state_count = payload.get("completed_count")
    if not isinstance(state_count, int):
        return None
    if state_count < 0 or state_count > completed_count:
        return None
    model.temporal_kg = payload["temporal_kg"]
    model.adaptive_fusion = payload["adaptive_fusion"]
    return state_count


def summarize(
    records: list[dict[str, Any]],
    started_at: float,
    output_path: Path,
    input_path: Path,
    resume_strategy: str,
) -> dict[str, Any]:
    ok_records = [record for record in records if record["status"] == "ok"]
    metrics = [record_metrics(record) for record in ok_records]

    def mean(key: str) -> float:
        return sum(metric[key] for metric in metrics) / len(metrics) if metrics else 0.0

    row_times = [record["elapsed_seconds"] for record in ok_records]
    return {
        "status": "ok" if len(ok_records) == len(records) else "partial",
        "input": str(input_path),
        "output": str(output_path),
        "requested_rows": len(records),
        "completed_rows": len(ok_records),
        "elapsed_seconds": time.time() - started_at,
        "avg_row_seconds": sum(row_times) / len(row_times) if row_times else 0.0,
        "max_context_size": max((record.get("context_size", 0) for record in ok_records), default=0),
        "resume_strategy": resume_strategy,
        "metrics": {
            "precision_at_10": mean("precision_at_10"),
            "recall_at_10": mean("recall_at_10"),
            "f1_at_10": mean("f1_at_10"),
            "hit_at_10": mean("hit_at_10"),
        },
        "notes": [
            "Context-equivalent checkpoint run: each row receives all previous row texts as LLM context.",
            "The execution order follows DKNet.run(documents), but predictions are written after each row.",
            "When --resume is used, completed rows are restored either from a matching state snapshot or by exact replay.",
            "Weak hashtags are proxy labels and not expert annotations.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-state-every", type=int, default=1)
    parser.add_argument("--no-state-resume", action="store_true")
    args = parser.parse_args()

    configure_paths()
    from src.dknet_model import DKNet

    rows = load_weak_label_rows(args.input, args.limit)
    documents = [make_document(row) for row in rows]
    model = DKNet(load_config())
    records = read_existing_records(args.output) if args.resume else []
    completed_count = len(records)
    resume_strategy = "fresh"
    if completed_count > len(rows):
        raise ValueError(f"Existing record count {completed_count} exceeds requested rows {len(rows)}")
    if completed_count:
        state_count = None if args.no_state_resume else try_load_state_prefix(model, args.state, completed_count)
        if state_count is None:
            replay_rows(model, documents, rows, completed_count)
            save_state(model, args.state, completed_count)
            resume_strategy = "exact_replay"
        else:
            if state_count < completed_count:
                replay_rows(model, documents, rows, completed_count, start=state_count)
                save_state(model, args.state, completed_count)
            resume_strategy = "state_snapshot"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and completed_count else "w"
    started_at = time.time()
    with args.output.open(mode, encoding="utf-8") as out_fh:
        for index, row in enumerate(rows[completed_count:], start=completed_count):
            row_started = time.time()
            record: dict[str, Any] = {
                "row_index": index + 1,
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
                prediction, stats = run_one_document(model, documents, index, row["_labels"])
                record["predicted"] = make_jsonable(prediction)
                record["stats"] = make_jsonable(stats)
                record["context_size"] = stats["context_size"]
            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
            record["elapsed_seconds"] = time.time() - row_started
            records.append(record)
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            if args.save_state_every > 0 and len(records) % args.save_state_every == 0:
                save_state(model, args.state, len(records))

    summary = summarize(records, started_at, args.output, args.input, resume_strategy)
    save_state(model, args.state, len(records))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
