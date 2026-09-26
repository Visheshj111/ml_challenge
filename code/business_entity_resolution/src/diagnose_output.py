"""Per-country sanity report for a matching_results.tsv file.

Training data only contains US and India while the public test set is
~47% India / ~38% US / ~15% France (zero-shot), so France is the first
suspect when the public score drops below validation. This script reports,
per country: entity counts, empty-prediction rate, and match-count
quantiles. Pass --matching-b to compare two submissions side by side.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_s1_countries(test_dir: str) -> pd.DataFrame:
    prefix = Path(test_dir).name
    path = os.path.join(test_dir, f"{prefix}_source1.tsv")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                        usecols=["entity_id", "country"])
    frame["country"] = frame["country"].astype(str).str.strip().str.lower()
    return frame


def country_stats(matching_path: str, s1: pd.DataFrame) -> pd.DataFrame:
    preds = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    preds["n_matches"] = preds["matched_entity_ids"].astype(str).str.split(",").map(
        lambda parts: len([p for p in parts if p.strip()])
    )
    merged = s1.merge(
        preds[["source1_entity_id", "n_matches"]],
        left_on="entity_id", right_on="source1_entity_id", how="left",
    )
    missing = int(merged["n_matches"].isna().sum())
    if missing:
        print(f"[diag] WARNING: {missing:,} S1 rows are missing from '{matching_path}'")
    merged = merged.dropna(subset=["n_matches"]).copy()
    merged["n_matches"] = merged["n_matches"].astype(int)
    rows = []
    for country, group in merged.groupby("country"):
        counts = group["n_matches"].to_numpy()
        rows.append({
            "country": country,
            "entities": len(group),
            "empty_rate": float((counts == 0).mean()),
            "mean_matches": float(counts.mean()),
            "median_matches": float(np.median(counts)),
            "p90_matches": float(np.quantile(counts, 0.9)),
            "multi_rate": float((counts >= 2).mean()),
        })
    return pd.DataFrame(rows).sort_values("entities", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--matching-b", default=None,
                        help="Second submission file for comparison (e.g. the 0.705 baseline)")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--model-meta", default=None,
                        help="Optional model_meta.json whose thresholds should be printed")
    args = parser.parse_args()

    if args.model_meta and os.path.isfile(args.model_meta):
        with open(args.model_meta, encoding="utf-8") as handle:
            meta = json.load(handle)
        print(f"[diag] {args.model_meta}: first={meta.get('first_threshold')} "
              f"additional={meta.get('additional_threshold')} val_f05={meta.get('val_f05')}")

    s1 = load_s1_countries(args.test_dir)
    for label, path in [("A (current)", args.matching), ("B (other)", args.matching_b)]:
        if path is None:
            continue
        if not os.path.isfile(path):
            print(f"[diag] SKIP {label}: '{path}' not found")
            continue
        print(f"\n[diag] ==== {label}: {path} ====")
        stats = country_stats(path, s1)
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
