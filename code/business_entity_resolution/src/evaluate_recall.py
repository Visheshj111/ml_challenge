"""Recall measurement for candidate retrieval, before any matcher training.

Measures, on a seeded sample of train S1 entities with known ground truth:

* lexical          - hash-index retrieval only (the baseline)
* dense            - multilingual embedding top-k only
* hybrid           - reciprocal-rank fusion of both pools, evaluated up to max K
* hybrid@<top_k>   - exactly the list production would ship (production pools)

Recall@K is micro-averaged over true match pairs (found / total): the ceiling
the downstream matcher can score. Use --no-dense for a lexical-only baseline,
and --cand-limit to compare encoders on a candidate subsample (recall then
drops artificially and is only comparable across encoders, not to a full run).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from blocking import load_sources
from blocking import retrieve_batch as lexical_retrieve_batch
from hybrid_blocking import (
    DEFAULT_ENCODE_BATCH,
    DEFAULT_ENCODER,
    HybridIndex,
    iter_country_batches,
    merge_hits,
)
from preprocess import prepare_record


def load_ground_truth(path: str) -> dict[str, set[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        s1_id: set(matches.split(",")) if matches else set()
        for s1_id, matches in frame[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False, name=None)
    }


def parse_ks(text: str) -> list[int]:
    return sorted({int(part) for part in text.split(",") if part.strip()})


def sample_s1(s1: pd.DataFrame, ground_truth: dict[str, set[str]], sample: int, seed: int) -> pd.DataFrame:
    mask = s1["entity_id"].map(lambda entity_id: bool(ground_truth.get(entity_id)))
    matchable = s1[mask]
    if sample and sample < len(matchable):
        positions = np.sort(np.random.default_rng(seed).choice(len(matchable), size=sample, replace=False))
        matchable = matchable.iloc[positions]
    print(f"[eval] Sampled {len(matchable):,} S1 entities with >=1 true match (of {int(mask.sum()):,} matchable)")
    return matchable.reset_index(drop=True)


def collect_records(
    s1_sample: pd.DataFrame,
    index: HybridIndex,
    ground_truth: dict[str, set[str]],
    candidate_ids: np.ndarray,
    ks: list[int],
    prod_lexical_k: int,
    prod_dense_k: int,
    prod_top_k: int,
    batch_size: int,
) -> tuple[list[tuple[str, set[str], dict[str, list[str]]]], float]:
    """One retrieval pass producing, per sampled entity, ranked ID lists per mode.

    The production hybrid mode is rebuilt from the production pool sizes
    (lexical_k / dense_k), not the larger evaluation pools, so its numbers
    reflect exactly what ``HybridIndex.retrieve_batch`` would ship.
    """
    max_k = max(ks)
    eval_lex_k = max(prod_lexical_k, max_k)
    eval_dense_k = max(prod_dense_k, max_k)
    records: list[tuple[str, set[str], dict[str, list[str]]]] = []
    started = time.time()
    processed = 0
    for batch in iter_country_batches(s1_sample, batch_size=batch_size):
        lexical = lexical_retrieve_batch(batch, index.lexical, top_k=eval_lex_k)
        dense_hits = index.dense.search(batch, eval_dense_k) if index.dense is not None else None
        for entity_id, name, address, country in batch[
            ["entity_id", "business_name", "business_address", "country"]
        ].itertuples(index=False, name=None):
            truth = ground_truth.get(entity_id, set())
            lex_hits = lexical.get(entity_id, [])
            modes: dict[str, list[str]] = {
                "lexical": [str(candidate_ids[hit.row]) for hit in lex_hits]
            }
            if dense_hits is not None:
                dense_list = dense_hits.get(entity_id, [])
                modes["dense"] = [str(candidate_ids[row]) for row, _score, _rank in dense_list]
                modes["hybrid"] = [
                    str(candidate_ids[hit.row]) for hit in merge_hits(lex_hits, dense_list, max_k)
                ]
                modes[f"hybrid@{prod_top_k}"] = [
                    str(candidate_ids[hit.row])
                    for hit in merge_hits(
                        lex_hits[:prod_lexical_k], dense_list[:prod_dense_k], prod_top_k
                    )
                ]
            records.append((prepare_record(name, address, country)["country"], truth, modes))
        processed += len(batch)
        print(f"[eval] Retrieved for {processed:,}/{len(s1_sample):,} sampled entities")
    return records, time.time() - started


def summarise(
    records: list[tuple[str, set[str], dict[str, list[str]]]], ks: list[int]
) -> dict[str, dict[str, object]]:
    names = list(records[0][2].keys()) if records else []
    summary: dict[str, dict[str, object]] = {}
    for name in names:
        found = {k: 0 for k in ks}
        total = 0
        candidates = 0
        for _country, truth, modes in records:
            ids = modes[name]
            candidates += len(ids)
            total += len(truth)
            for k in ks:
                prefix = ids[:k]
                found[k] += sum(1 for truth_id in truth if truth_id in prefix)
        summary[name] = {
            "mean_candidates": round(candidates / len(records), 1) if records else 0.0,
            "recall": {k: (found[k] / total if total else 0.0) for k in ks},
        }
    return summary


def per_country_stats(
    records: list[tuple[str, set[str], dict[str, list[str]]]], mode: str, k: int
) -> dict[str, tuple[int, int, int]]:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # found, truth, entities
    for country, truth, modes in records:
        ids = modes[mode][:k]
        entry = stats[country]
        entry[0] += sum(1 for truth_id in truth if truth_id in ids)
        entry[1] += len(truth)
        entry[2] += 1
    return {country: (found, total, entities) for country, (found, total, entities) in stats.items()}


def print_table(summary: dict[str, dict[str, object]], ks: list[int]) -> None:
    print(f"\n{'mode':<22}{'cands/entity':>14}" + "".join(f"{'R@' + str(k):>9}" for k in ks))
    for name, stats in summary.items():
        row = f"{name:<22}{float(stats['mean_candidates']):>14.1f}"
        row += "".join(f"{stats['recall'][k]:>9.4f}" for k in ks)
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure candidate recall@K for lexical / dense / hybrid retrieval."
    )
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--sample", type=int, default=3000, help="Sampled S1 entities with >=1 true match")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", default="10,25,50,60,100")
    parser.add_argument("--top-k", type=int, default=60, help="Production merged candidate count")
    parser.add_argument("--lexical-k", type=int, default=100, help="Production lexical pool size")
    parser.add_argument("--dense-k", type=int, default=40, help="Production dense pool size")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch", type=int, default=DEFAULT_ENCODE_BATCH)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--cand-limit", type=int, default=None,
                        help="Truncate candidates for fast encoder comparison (breaks absolute recall)")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--save", default=None, help="Optional path to write the results as JSON")
    args = parser.parse_args()

    ks = parse_ks(args.ks)
    s1, s2, s3 = load_sources(args.data_dir)
    candidates = pd.concat([s2, s3], ignore_index=True)
    if args.cand_limit is not None:
        candidates = candidates.head(args.cand_limit).copy()
        print(f"[eval] cand_limit={args.cand_limit}: recall is NOT comparable to a full run")
    candidate_ids = candidates["entity_id"].to_numpy(copy=False)
    ground_truth = load_ground_truth(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    sample = sample_s1(s1, ground_truth, args.sample, args.seed)

    build_started = time.time()
    index = HybridIndex.build(
        candidates, args.data_dir, model_dir=args.model_dir, top_k=args.top_k,
        lexical_k=args.lexical_k, dense_k=args.dense_k, encoder=args.encoder,
        device=args.device, encode_batch=args.encode_batch,
        dense_enabled=not args.no_dense, cand_limit=args.cand_limit,
    )
    build_seconds = time.time() - build_started

    records, retrieval_seconds = collect_records(
        sample, index, ground_truth, candidate_ids, ks,
        args.lexical_k, args.dense_k, args.top_k, args.batch_size,
    )
    if not records:
        print("[eval] No sampled entities had retrievable candidates; nothing to report.")
        return 1

    summary = summarise(records, ks)
    print(f"\n[eval] candidates={len(candidates):,}; sampled entities={len(sample):,}; "
          f"index build={build_seconds:,.0f}s; retrieval={retrieval_seconds:,.0f}s")
    print_table(summary, ks)

    report_k = min(args.top_k, max(ks))
    prod_mode = f"hybrid@{args.top_k}"
    columns = ["lexical", "dense", "hybrid", prod_mode] if index.dense is not None else ["lexical"]
    stats = {mode: per_country_stats(records, mode, report_k) for mode in columns}
    print(f"\nPer-country recall@{report_k} (country of the S1 entity):")
    print(f"{'country':<18}{'entities':>10}{'truth':>8}" + "".join(f"{name[:12]:>14}" for name in columns))
    base = stats[columns[0]]
    for country in sorted(set().union(*(set(mode_stats) for mode_stats in stats.values()))):
        _found, total, entities = base.get(country, (0, 0, 0))
        row = f"{country or '<empty>':<18}{entities:>10}{total:>8}"
        for mode in columns:
            found = stats[mode].get(country, (0, 0, 0))[0]
            row += f"{(found / total if total else 0.0):>14.4f}"
        print(row)

    if args.save:
        payload = {
            "data_dir": args.data_dir,
            "encoder": index.dense.encoder if index.dense is not None else None,
            "cand_limit": args.cand_limit,
            "sample": len(sample),
            "seed": args.seed,
            "ks": ks,
            "top_k": args.top_k,
            "lexical_k": args.lexical_k,
            "dense_k": args.dense_k,
            "build_seconds": round(build_seconds, 1),
            "retrieval_seconds": round(retrieval_seconds, 1),
            "summary": summary,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n[eval] Saved results to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
