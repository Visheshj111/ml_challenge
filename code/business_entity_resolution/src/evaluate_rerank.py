"""Fresh-slice evaluation of cross-encoder reranking + decision-rule tuning.

Scores the SAME seed-42 slice as tune_decisions.py (disjoint from the
LightGBM training rows order[10000:60000] and validation rows
order[:10000]), then reports:

  1. lgbm_full      - LightGBM-only sweep; must reproduce tune_decisions'
                      ~0.7508 as a plumbing check
  2. lgbm_top_n     - the F0.5 cost of only being allowed to predict from
                      each entity's N best LightGBM-ranked pairs (the rerank
                      restriction tax, known BEFORE training anything)
  3. ce_rerank      - sweep after cross-encoder reranking of those top-N
                      pairs, on a wide grid (CE probabilities live below 0.9)
  4. per-country rules using the ACTUAL country labels. tune_decisions
     compared 'US'/'India' values against lowercase literals, so its
     per-country coordinate descent silently never ran (see the empty
     '[]' list in its log); that question is still open.

Run without --cross-encoder for references 1+2 only. Writes
rerank_calibration.json next to the model; predict.py consumes it via
--decision-rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import load_sources
from features import extract_features_for_hits
from hybrid_blocking import HybridIndex, iter_country_batches, retrieve_batch
from rerank import build_candidate_texts, build_s1_texts, load_cross_encoder, rerank_batch_probabilities
from scorer import compute_validation_f05
from tune_decisions import _build_groups, _predict_with_rules

SEED = 42
VALIDATION_COUNT = 10_000
TRAIN_COUNT = 50_000
SLICE_START = VALIDATION_COUNT + TRAIN_COUNT
SLICE_COUNT = 25_000

LGBM_GRID = np.round(np.arange(0.85, 0.9951, 0.005), 4)
CE_COARSE = np.round(np.arange(0.30, 0.9951, 0.02), 4)
CE_REFINE_STEP = 0.005
CE_REFINE_SPAN = 0.03
CAP_CHOICES = (1, 2, 3, 5, 8)


def _score(groups, s1_countries, rules, ground_truth, cal_ids) -> float:
    predictions = _predict_with_rules(groups, s1_countries, rules)
    return compute_validation_f05(predictions, ground_truth, cal_ids)


def _sweep_global(groups, s1_countries, ground_truth, cal_ids, first_grid, add_grid, cap, best):
    """best: (first, additional, cap, f05); returns an improved or equal best."""
    for first in first_grid:
        for add in add_grid[add_grid >= first]:
            rules = {"*": (float(first), float(add), cap)}
            score = _score(groups, s1_countries, rules, ground_truth, cal_ids)
            if score > best[3]:
                best = (float(first), float(add), cap, float(score))
    return best


def _sweep_caps(groups, s1_countries, ground_truth, cal_ids, base):
    best = base
    best_cap = base[2]
    for cap in CAP_CHOICES:
        rules = {"*": (base[0], base[1], cap)}
        score = _score(groups, s1_countries, rules, ground_truth, cal_ids)
        if score > best[3]:
            best = (base[0], base[1], cap, float(score))
            best_cap = cap
    return best, best_cap


def _refine_grid(value: float) -> np.ndarray:
    low = max(0.05, value - CE_REFINE_SPAN)
    high = min(0.995, value + CE_REFINE_SPAN + 1e-9)
    return np.round(np.arange(low, high, CE_REFINE_STEP), 4)


def _sweep_country(groups, s1_countries, ground_truth, cal_ids, country, base_rules, grid):
    """One coordinate-descent pass over one country's (first, additional)."""
    best_score = _score(groups, s1_countries, base_rules, ground_truth, cal_ids)
    best_rules = dict(base_rules)
    cap = base_rules["*"][2]
    for first in grid:
        for add in grid[grid >= first]:
            trial = dict(best_rules)
            trial[country] = (float(first), float(add), cap)
            score = _score(groups, s1_countries, trial, ground_truth, cal_ids)
            if score > best_score:
                best_score = score
                best_rules = trial
    return best_rules, best_score


def _restrict_groups(groups, top_n: int) -> dict:
    return {sid: (scores[:top_n], ids[:top_n]) for sid, (scores, ids) in groups.items()}


def _per_country_breakdown(groups, s1_countries, rules, ground_truth, cal_ids) -> dict:
    predictions = _predict_with_rules(groups, s1_countries, rules)
    by_country: dict[str, list[str]] = {}
    for sid in cal_ids:
        by_country.setdefault(s1_countries.get(sid, "?"), []).append(sid)
    return {
        country: compute_validation_f05(predictions, ground_truth, sids)
        for country, sids in sorted(by_country.items())
    }


def _serialise_rules(rules) -> dict:
    return {c: {"first": r[0], "additional": r[1], "cap": r[2]} for c, r in rules.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--gt-path", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--model-dir", default="models/hybrid_v1")
    parser.add_argument("--cross-encoder", default=None,
                        help="Fine-tuned model dir or HF checkpoint; omit for LightGBM-only references")
    parser.add_argument("--ce-top-n", type=int, default=10)
    parser.add_argument("--ce-batch-size", type=int, default=256)
    parser.add_argument("--slice-count", type=int, default=SLICE_COUNT)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None,
                        help="Output JSON (default: <model-dir>/rerank_calibration.json)")
    args = parser.parse_args()

    meta_path = os.path.join(args.model_dir, "model_meta.json")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    model = joblib.load(os.path.join(args.model_dir, "matcher_model.joblib"))
    top_k = int(meta["top_k_blocking"])
    out_path = args.out or os.path.join(args.model_dir, "rerank_calibration.json")

    s1, s2, s3 = load_sources(args.train_dir)
    candidates = pd.concat([s2, s3], ignore_index=True)
    entity_ids = candidates["entity_id"].to_numpy(copy=False)

    gt_df = pd.read_csv(args.gt_path, sep="\t", dtype=str).fillna("")
    ground_truth = {
        sid: (set(m.split(",")) if m.strip() else set())
        for sid, m in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"])
    }

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(s1))
    slice_rows = order[SLICE_START:SLICE_START + args.slice_count]
    s1_cal = s1.iloc[slice_rows].reset_index(drop=True)
    cal_ids = s1_cal["entity_id"].tolist()
    s1_countries = {sid: str(c).strip().lower() for sid, c in zip(s1_cal["entity_id"], s1_cal["country"])}
    n_singletons = sum(1 for sid in cal_ids if not ground_truth.get(sid))
    mix = s1_cal["country"].astype(str).value_counts(normalize=True).round(4).to_dict()
    print(f"[rerank-eval] Slice rows={len(s1_cal):,} | country mix={mix} | "
          f"singletons={n_singletons:,} ({n_singletons / len(cal_ids):.1%})")

    index = HybridIndex.build(
        candidates, args.train_dir, model_dir=args.model_dir, top_k=top_k,
        lexical_k=int(meta["lexical_k"]), dense_k=int(meta["dense_k"]),
        encoder=meta["encoder"], device=args.device,
        encode_batch=int(meta["encode_batch"]), dense_enabled=bool(meta["dense_enabled"]),
    )

    ce = load_cross_encoder(args.cross_encoder, device=args.device) if args.cross_encoder else None
    cand_texts = build_candidate_texts(candidates) if ce is not None else None

    lgbm_chunks, ce_chunks, sid_chunks, row_chunks = [], [], [], []
    done = 0
    for batch in iter_country_batches(s1_cal, batch_size=args.batch_size):
        hits_by_s1 = retrieve_batch(batch, index, top_k=top_k)
        X, source_ids, candidate_rows, _hits = extract_features_for_hits(batch, candidates, hits_by_s1)
        lgbm_probs = model.predict_proba(X)[:, 1] if len(X) else np.empty(0, dtype=np.float32)
        lgbm_chunks.append(np.asarray(lgbm_probs, dtype=np.float64))
        sid_chunks.extend(source_ids)
        row_chunks.extend(candidate_rows)
        if ce is not None:
            ce_probs = rerank_batch_probabilities(
                build_s1_texts(batch), cand_texts, source_ids, candidate_rows,
                lgbm_probs, ce, top_n=args.ce_top_n, ce_batch_size=args.ce_batch_size,
            )
            ce_chunks.append(ce_probs)
        done += len(batch)
        print(f"[rerank-eval] Scored {done:,}/{len(s1_cal):,} S1 rows")

    lgbm_probs = np.concatenate(lgbm_chunks) if lgbm_chunks else np.empty(0)
    groups_lgbm = _build_groups(sid_chunks, row_chunks, lgbm_probs, entity_ids)

    results = {
        "slice": {"rows": len(cal_ids), "country_mix": mix,
                  "singleton_rate": n_singletons / len(cal_ids)},
        "ce": ({"model": args.cross_encoder, "top_n": args.ce_top_n} if ce is not None else None),
    }

    print("\n[rerank-eval] Reference 1: LightGBM full candidate sweep ...")
    ref = (float(meta["first_threshold"]), float(meta["additional_threshold"]), None, -1.0)
    ref = _sweep_global(groups_lgbm, s1_countries, ground_truth, cal_ids, LGBM_GRID, LGBM_GRID, None, ref)
    ref, ref_cap = _sweep_caps(groups_lgbm, s1_countries, ground_truth, cal_ids, ref)
    print(f"[rerank-eval]   lgbm_full best: first={ref[0]:.3f} add={ref[1]:.3f} "
          f"cap={ref_cap} F0.5={ref[3]:.4f}")
    results["lgbm_full"] = {"first": ref[0], "additional": ref[1], "cap": ref_cap, "f05": ref[3]}

    print("[rerank-eval] Reference 2: LightGBM restricted to top-"
          f"{args.ce_top_n} pairs per entity ...")
    groups_topn = _restrict_groups(groups_lgbm, args.ce_top_n)
    ref2 = (ref[0], ref[1], None, -1.0)
    ref2 = _sweep_global(groups_topn, s1_countries, ground_truth, cal_ids, LGBM_GRID, LGBM_GRID, None, ref2)
    ref2, ref2_cap = _sweep_caps(groups_topn, s1_countries, ground_truth, cal_ids, ref2)
    tax = ref[3] - ref2[3]
    print(f"[rerank-eval]   lgbm_top_{args.ce_top_n} best: first={ref2[0]:.3f} add={ref2[1]:.3f} "
          f"cap={ref2_cap} F0.5={ref2[3]:.4f} (restriction tax {tax:+.4f})")
    results["lgbm_top_n"] = {"top_n": args.ce_top_n, "first": ref2[0], "additional": ref2[1],
                             "cap": ref2_cap, "f05": ref2[3], "restriction_tax": tax}

    best_rules = {"*": (ref[0], ref[1], ref_cap)}
    best_score = ref[3]
    best_source = "lgbm_full"

    if ce is not None:
        ce_probs = np.concatenate(ce_chunks) if ce_chunks else np.empty(0)
        groups_ce = _build_groups(sid_chunks, row_chunks, ce_probs, entity_ids)

        print("\n[rerank-eval] Cross-encoder rerank sweep (coarse) ...")
        ce_best = (0.5, 0.5, None, -1.0)
        ce_best = _sweep_global(groups_ce, s1_countries, ground_truth, cal_ids,
                                CE_COARSE, CE_COARSE, None, ce_best)
        print(f"[rerank-eval]   coarse best: first={ce_best[0]:.3f} add={ce_best[1]:.3f} F0.5={ce_best[3]:.4f}")
        print("[rerank-eval] Refining around the coarse best ...")
        ce_best = _sweep_global(groups_ce, s1_countries, ground_truth, cal_ids,
                                _refine_grid(ce_best[0]), _refine_grid(ce_best[1]), None, ce_best)
        print(f"[rerank-eval]   refined best: first={ce_best[0]:.3f} add={ce_best[1]:.3f} F0.5={ce_best[3]:.4f}")
        ce_best, ce_cap = _sweep_caps(groups_ce, s1_countries, ground_truth, cal_ids, ce_best)
        print(f"[rerank-eval]   with cap={ce_cap}: F0.5={ce_best[3]:.4f}")
        results["ce_rerank_global"] = {"first": ce_best[0], "additional": ce_best[1],
                                       "cap": ce_cap, "f05": ce_best[3]}

        print("[rerank-eval] Per-country refinement (labels lowercased, matching predict.py) ...")
        ce_rules = {"*": (ce_best[0], ce_best[1], ce_cap)}
        ce_score = ce_best[3]
        for country in sorted({c for c in s1_countries.values() if c}):
            ce_rules, ce_score = _sweep_country(
                groups_ce, s1_countries, ground_truth, cal_ids, country, ce_rules, CE_COARSE
            )
            rule = ce_rules.get(country) or ce_rules["*"]
            print(f"[rerank-eval]   {country}: rule=({rule[0]:.3f},{rule[1]:.3f},cap{rule[2]}) "
                  f"F0.5={ce_score:.4f}")
        for cap in CAP_CHOICES + (None,):
            trial = {c: (r[0], r[1], cap) for c, r in ce_rules.items()}
            score = _score(groups_ce, s1_countries, trial, ground_truth, cal_ids)
            if score > ce_score:
                ce_score = score
                ce_rules = trial
        results["ce_rerank"] = {"rules": _serialise_rules(ce_rules), "f05": ce_score}

        if ce_score > best_score:
            best_rules, best_score, best_source = ce_rules, ce_score, "ce_rerank"
            print(f"\n[rerank-eval] Cross-encoder WINS: F0.5={ce_score:.4f} "
                  f"(vs lgbm_full {ref[3]:.4f}, gain {ce_score - ref[3]:+.4f})")
        else:
            print(f"\n[rerank-eval] Cross-encoder LOSES: {ce_score:.4f} vs lgbm_full {ref[3]:.4f} "
                  "- do NOT ship the reranker.")

    breakdown = _per_country_breakdown(
        groups_ce if (ce is not None and best_source == "ce_rerank") else groups_lgbm,
        s1_countries, best_rules, ground_truth, cal_ids,
    )
    results["per_country_breakdown"] = breakdown
    results["recommended"] = {
        "rules": _serialise_rules(best_rules),
        "fresh_slice_f05": best_score,
        "source": best_source,
        "ce_top_n": args.ce_top_n if best_source == "ce_rerank" else None,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n[rerank-eval] ==== SUMMARY ====")
    print(f"[rerank-eval] lgbm_full:        F0.5={ref[3]:.4f}")
    print(f"[rerank-eval] lgbm_top_{args.ce_top_n}:      F0.5={ref2[3]:.4f} (tax {tax:+.4f})")
    if ce is not None:
        print(f"[rerank-eval] ce_rerank:       F0.5={results['ce_rerank']['f05']:.4f}")
    print(f"[rerank-eval] Recommended ({best_source}): {results['recommended']['rules']}")
    print(f"[rerank-eval] Per-country F0.5: {breakdown}")
    print(f"[rerank-eval] Full results: {out_path}")
    print(f"[rerank-eval] predict.py usage: --decision-rules {out_path}"
          + (f" --cross-encoder {args.cross_encoder} --ce-top-n {args.ce_top_n}"
             if best_source == "ce_rerank" else ""))


if __name__ == "__main__":
    main()
