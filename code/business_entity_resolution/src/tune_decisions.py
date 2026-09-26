"""Fresh-slice decision-rule calibration for the hybrid matcher.

The shipped thresholds (0.95, 0.95) sit at the optimiser grid corner, so this
script scores a calibration slice that is disjoint from BOTH the training rows
(order[10000:60000]) and the threshold-tuning validation rows (order[:10000])
under the same seed-42 permutation train_model.py uses, then searches:

  1. shipped rule (0.95, 0.95)  - reference point
  2. fine global (first, additional) grid up to 0.995
  3. top-N caps                 - macro F0.5 punishes over-prediction hard
  4. per-country (india/us) thresholds via coordinate descent

Read-only on the model and cache. Writes models/hybrid_v1/decision_calibration.json.
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
from scorer import compute_validation_f05
from train_model import _decision_predictions

SEED = 42
VALIDATION_COUNT = 10_000
TRAIN_COUNT = 50_000
SLICE_START = VALIDATION_COUNT + TRAIN_COUNT
SLICE_COUNT = 25_000

GRID = np.round(np.arange(0.90, 0.9951, 0.005), 4)
COUNTRY_GRID = np.round(np.arange(0.90, 0.9951, 0.01), 4)
CAP_CHOICES = (1, 2, 3, 5)


def _build_groups(source_ids, candidate_rows, probabilities, entity_ids):
    """Per-s1 sorted (score desc, row desc) groups mirroring _decision_predictions."""
    grouped = {}
    for sid, row, p in zip(source_ids, candidate_rows, probabilities):
        grouped.setdefault(sid, []).append((float(p), int(row)))
    groups = {}
    for sid, pairs in grouped.items():
        pairs.sort(reverse=True)
        scores = np.fromiter((p for p, _ in pairs), dtype=np.float64, count=len(pairs))
        ids = [str(entity_ids[row]) for _, row in pairs]
        groups[sid] = (scores, ids)
    return groups


def _predict_with_rules(groups, s1_countries, rules):
    """rules: country -> (first, additional, cap); '*' is the fallback."""
    predictions = {}
    for sid, (scores, ids) in groups.items():
        first, add, cap = rules.get(s1_countries.get(sid)) or rules["*"]
        if scores[0] < first:
            predictions[sid] = []
            continue
        k = int(np.count_nonzero(scores >= add))
        if cap is not None:
            k = min(k, cap)
        accepted = ids[:k]
        if ids[0] not in accepted:
            accepted = ([ids[0]] + accepted)[:cap] if cap else [ids[0]] + accepted
        predictions[sid] = accepted
    return predictions


def _evaluate(groups, s1_countries, rules, ground_truth, cal_ids):
    predictions = _predict_with_rules(groups, s1_countries, rules)
    return compute_validation_f05(predictions, ground_truth, cal_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--gt-path", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--model-dir", default="models/hybrid_v1")
    parser.add_argument("--slice-count", type=int, default=SLICE_COUNT)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    meta_path = os.path.join(args.model_dir, "model_meta.json")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    model = joblib.load(os.path.join(args.model_dir, "matcher_model.joblib"))
    top_k = int(meta["top_k_blocking"])

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
    s1_countries = {sid: str(c) for sid, c in zip(s1_cal["entity_id"], s1_cal["country"])}
    n_singletons = sum(1 for sid in cal_ids if not ground_truth.get(sid))
    mix = s1_cal["country"].astype(str).value_counts(normalize=True).round(4).to_dict()
    print(f"[cal] Slice rows={len(s1_cal):,} | country mix={mix} | "
          f"singletons={n_singletons:,} ({n_singletons / len(cal_ids):.1%})")

    index = HybridIndex.build(
        candidates, args.train_dir, model_dir=args.model_dir, top_k=top_k,
        lexical_k=int(meta["lexical_k"]), dense_k=int(meta["dense_k"]),
        encoder=meta["encoder"], device=args.device,
        encode_batch=int(meta["encode_batch"]), dense_enabled=bool(meta["dense_enabled"]),
    )

    prob_chunks, sid_chunks, row_chunks = [], [], []
    done = 0
    for batch in iter_country_batches(s1_cal, batch_size=args.batch_size):
        hits_by_s1 = retrieve_batch(batch, index, top_k=top_k)
        X, source_ids, candidate_rows, _hits = extract_features_for_hits(batch, candidates, hits_by_s1)
        probs = model.predict_proba(X)[:, 1] if len(X) else np.empty(0, dtype=np.float32)
        prob_chunks.append(probs.astype(np.float64))
        sid_chunks.extend(source_ids)
        row_chunks.extend(candidate_rows)
        done += len(batch)
        print(f"[cal] Scored {done:,}/{len(s1_cal):,} S1 rows")
    probabilities = np.concatenate(prob_chunks)

    groups = _build_groups(sid_chunks, row_chunks, probabilities, entity_ids)

    shipped_first = float(meta["first_threshold"])
    shipped_add = float(meta["additional_threshold"])
    shipped_preds = _decision_predictions(
        sid_chunks, row_chunks, probabilities, entity_ids, shipped_first, shipped_add
    )
    shipped_f05 = compute_validation_f05(shipped_preds, ground_truth, cal_ids)
    print(f"[cal] Shipped rule ({shipped_first:.2f},{shipped_add:.2f}) fresh-slice macro F0.5 = {shipped_f05:.4f}")

    results = {
        "slice": {"rows": len(cal_ids), "country_mix": mix, "singleton_rate": n_singletons / len(cal_ids)},
        "shipped": {"first": shipped_first, "additional": shipped_add, "f05": shipped_f05},
        "global_grid": [], "caps": [], "per_country": [],
    }

    print("[cal] Sweeping fine global threshold grid ...")
    best_global = (shipped_first, shipped_add, None, shipped_f05)
    for first in GRID:
        for add in GRID[GRID >= first]:
            rules = {"*": (float(first), float(add), None)}
            score = _evaluate(groups, s1_countries, rules, ground_truth, cal_ids)
            results["global_grid"].append({"first": float(first), "additional": float(add), "f05": score})
            if score > best_global[3]:
                best_global = (float(first), float(add), None, score)
    print(f"[cal] Best global: first={best_global[0]:.3f} additional={best_global[1]:.3f} "
          f"F0.5={best_global[3]:.4f} (gain {best_global[3] - shipped_f05:+.4f})")

    print("[cal] Sweeping top-N caps at best global thresholds ...")
    best_cap = None
    best_with_cap = best_global
    for cap in CAP_CHOICES:
        rules = {"*": (best_global[0], best_global[1], cap)}
        score = _evaluate(groups, s1_countries, rules, ground_truth, cal_ids)
        results["caps"].append({"cap": cap, "f05": score})
        print(f"[cal]   cap={cap}: F0.5={score:.4f}")
        if score > best_with_cap[3]:
            best_cap = cap
            best_with_cap = (best_global[0], best_global[1], cap, score)
    rules = {"*": (best_global[0], best_global[1], best_cap)}
    score = _evaluate(groups, s1_countries, rules, ground_truth, cal_ids)
    results["caps"].append({"cap": None, "f05": score})

    present_countries = sorted({c for c in s1_countries.values() if c in ("india", "us")})
    print(f"[cal] Coordinate descent on per-country thresholds for {present_countries} ...")
    best_rules = {"*": (best_with_cap[0], best_with_cap[1], best_with_cap[2])}
    best_score = best_with_cap[3]
    for _pass in range(2):
        for country in present_countries:
            for first in COUNTRY_GRID:
                for add in COUNTRY_GRID[COUNTRY_GRID >= first]:
                    trial = dict(best_rules)
                    trial[country] = (float(first), float(add), best_with_cap[2])
                    score = _evaluate(groups, s1_countries, trial, ground_truth, cal_ids)
                    results["per_country"].append(
                        {"pass": _pass, "country": country, "first": float(first),
                         "additional": float(add), "f05": score}
                    )
                    if score > best_score:
                        best_score = score
                        best_rules = trial
            c_rule = best_rules.get(country) or best_rules["*"]
            print(f"[cal]   pass {_pass} {country}: rule=({c_rule[0]:.2f},{c_rule[1]:.2f},cap{c_rule[2]}) "
                  f"F0.5={best_score:.4f}")

    print("[cal] Re-checking caps under per-country rules ...")
    for cap in CAP_CHOICES + (None,):
        trial = {c: (r[0], r[1], cap) for c, r in best_rules.items()}
        score = _evaluate(groups, s1_countries, trial, ground_truth, cal_ids)
        results["per_country"].append({"cap_recheck": cap, "f05": score})
        if score > best_score:
            best_score = score
            best_rules = trial

    gain = best_score - shipped_f05
    serialisable_rules = {
        c: {"first": r[0], "additional": r[1], "cap": r[2]} for c, r in best_rules.items()
    }
    results["recommended"] = {
        "rules": serialisable_rules,
        "fresh_slice_f05": best_score,
        "shipped_f05": shipped_f05,
        "gain": gain,
        "worth_overnight_rerun": bool(gain >= 0.005),
    }
    out_path = os.path.join(args.model_dir, "decision_calibration.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n[cal] ==== SUMMARY ====")
    print(f"[cal] Shipped ({shipped_first:.2f},{shipped_add:.2f}):        F0.5={shipped_f05:.4f}")
    print(f"[cal] Best global ({best_global[0]:.3f},{best_global[1]:.3f}):  F0.5={best_global[3]:.4f}")
    print(f"[cal] Best overall: {serialisable_rules}")
    print(f"[cal] Fresh-slice F0.5={best_score:.4f} (gain {gain:+.4f})")
    if gain >= 0.005:
        print("[cal] Gain >= +0.005 -> overnight predict.py re-run is justified.")
    else:
        print("[cal] Gain < +0.005 -> decision-rule recalibration alone is NOT worth a 5h re-run.")
    print(f"[cal] Full results: {out_path}")


if __name__ == "__main__":
    main()
