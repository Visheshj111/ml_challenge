"""Quantify cross-script ground-truth pairs (Latin S1 name vs Indic-script match).

The miss diagnostic showed hard_miss pairs dominated by script mismatch: the
S1 record's name is Latin ('Star Systems Private Limited') while the true
match is Devanagari/Gurmukhi/Telugu ('...'). Every lexical key in blocking.py
is an exact hash over normalised tokens, so such pairs share zero tokens and
can only be found by the dense encoder. This script measures how many true
pairs are cross-script -- the structural recall penalty no matcher can undo.

Read-only; needs no index, no GPU, no embedding cache (~3 min, ~3 GB RAM).
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

# Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
INDIC = re.compile(r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"
                   r"\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]")


def script_label(name: str) -> str:
    return "indic" if INDIC.search(name or "") else "latin"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset/train")
    args = parser.parse_args()

    read_kwargs = {"sep": "\t", "dtype": str, "keep_default_na": False}
    s1 = pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"),
                     usecols=["entity_id", "business_name", "country"], **read_kwargs)
    s2 = pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"),
                     usecols=["entity_id", "business_name"], **read_kwargs)
    s3 = pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"),
                     usecols=["entity_id", "business_name"], **read_kwargs)
    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), **read_kwargs)

    s1_script = pd.Series(
        [script_label(n) for n in s1["business_name"]], index=s1["entity_id"]
    )
    s1_country = s1.set_index("entity_id")["country"]
    cand_script = pd.Series(
        [script_label(n) for n in s2["business_name"]] +
        [script_label(n) for n in s3["business_name"]],
        index=pd.concat([s2["entity_id"], s3["entity_id"]], ignore_index=True),
    )

    pairs = gt[gt["matched_entity_ids"].str.strip() != ""].copy()
    pairs["match_id"] = pairs["matched_entity_ids"].str.split(",")
    pairs = pairs[["source1_entity_id", "match_id"]].explode("match_id")
    pairs["match_id"] = pairs["match_id"].str.strip()
    pairs = pairs[pairs["match_id"] != ""]

    pairs["s1_script"] = pairs["source1_entity_id"].map(s1_script)
    pairs["cand_script"] = pairs["match_id"].map(cand_script)
    pairs["country"] = pairs["source1_entity_id"].map(s1_country).str.strip().str.lower()
    known = pairs.dropna(subset=["s1_script", "cand_script"])

    total = len(known)
    cross = known[known["s1_script"] != known["cand_script"]]
    latin_s1 = known[known["s1_script"] == "latin"]
    cross_latin = cross[cross["s1_script"] == "latin"]

    print(f"[script] true pairs={total:,}")
    print(f"[script] cross-script pairs={len(cross):,} ({len(cross)/total:.1%} of all)")
    print(f"[script] latin-S1 pairs={len(latin_s1):,}; of these cross-script="
          f"{len(cross_latin):,} ({len(cross_latin)/max(len(latin_s1),1):.1%}) "
          f"-> invisible to every exact-token lexical key")
    print("[script] cross-script pairs by S1 country:")
    for country, group in cross.groupby("country"):
        country_total = (known["country"] == country).sum()
        print(f"  {country or '<empty>':<10} cross={len(group):>9,}  "
              f"of country pairs={country_total:>9,}  ({len(group)/max(country_total,1):.1%})")
    print("[script] lower-bound recall ceiling cost (assume ~all cross pairs missed): "
          f"-{len(cross)/total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
