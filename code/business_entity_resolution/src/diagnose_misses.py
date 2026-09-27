"""Diagnose WHICH true matches retrieval misses, and why.

On a seeded sample of train S1 entities with known ground truth, retrieve deep
pools (lexical top-``--lex-deep``, dense top-``--dense-deep``) and the shipped
production list (merge of lexical top-``--prod-lex`` + dense top-``--prod-dense``
into top-``--prod-top``). For every true match NOT in the production list, classify:

* ``pool_but_cut``    - present in a production pool but cut by the merge / top-k.
                        Fixable by raising top_k or tuning the fusion (cheap).
* ``ranked_too_deep`` - ranked inside the deep pools but far beyond production
                        pools. Needs much larger k or better scoring.
* ``hard_miss``       - absent from both deep pools. A genuine retrieval failure;
                        concrete examples are printed so the pattern is visible.

Also reports the miss rate per country to expose zero-shot (France) weakness.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from blocking import load_sources
from blocking import retrieve_batch as lexical_retrieve_batch
from hybrid_blocking import HybridIndex, iter_country_batches, merge_hits
from preprocess import prepare_record


def load_ground_truth(path: str) -> dict[str, set[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        s1_id: set(matches.split(",")) if matches else set()
        for s1_id, matches in frame[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False, name=None)
    }


def sample_s1(s1: pd.DataFrame, ground_truth: dict[str, set[str]], sample: int, seed: int) -> pd.DataFrame:
    mask = s1["entity_id"].map(lambda entity_id: bool(ground_truth.get(entity_id)))
    matchable = s1[mask]
    if sample and sample < len(matchable):
        positions = np.sort(np.random.default_rng(seed).choice(len(matchable), size=sample, replace=False))
        matchable = matchable.iloc[positions]
    print(f"[miss] Sampled {len(matchable):,} S1 entities with >=1 true match")
    return matchable.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show and categorise the true matches retrieval misses.")
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lex-deep", type=int, default=500)
    parser.add_argument("--dense-deep", type=int, default=200)
    parser.add_argument("--prod-lex", type=int, default=100)
    parser.add_argument("--prod-dense", type=int, default=40)
    parser.add_argument("--prod-top", type=int, default=60)
    parser.add_argument("--examples", type=int, default=25, help="How many hard_miss pairs to print")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    s1, s2, s3 = load_sources(args.data_dir)
    candidates = pd.concat([s2, s3], ignore_index=True)
    candidate_ids = candidates["entity_id"].to_numpy(copy=False)
    cand_name = candidates["business_name"].to_numpy(copy=False)
    cand_addr = candidates["business_address"].to_numpy(copy=False)
    pos_of = {eid: i for i, eid in enumerate(candidate_ids)}
    ground_truth = load_ground_truth(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    sample = sample_s1(s1, ground_truth, args.sample, args.seed)

    index = HybridIndex.build(
        candidates, args.data_dir, model_dir=args.model_dir, top_k=args.prod_top,
        lexical_k=args.prod_lex, dense_k=args.prod_dense,
    )

    total = missed = 0
    cats = Counter()
    by_country: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [total, missed]
    examples: list[tuple] = []

    for batch in iter_country_batches(sample, batch_size=args.batch_size):
        lex_deep = lexical_retrieve_batch(batch, index.lexical, top_k=args.lex_deep)
        dense_deep = index.dense.search(batch, args.dense_deep) if index.dense is not None else {}
        for entity_id, name, address, country in batch[
            ["entity_id", "business_name", "business_address", "country"]
        ].itertuples(index=False, name=None):
            truth = ground_truth.get(entity_id, set())
            if not truth:
                continue
            lex_list = lex_deep.get(entity_id, [])
            dense_list = dense_deep.get(entity_id, [])
            lex_rank = {candidate_ids[h.row]: i + 1 for i, h in enumerate(lex_list)}
            dense_rank = {candidate_ids[row]: i + 1 for i, (row, _s, _r) in enumerate(dense_list)}
            merged = merge_hits(lex_list[: args.prod_lex], dense_list[: args.prod_dense], args.prod_top)
            merged_ids = {candidate_ids[h.row] for h in merged}
            ctry = prepare_record(name, address, country)["country"]

            for tid in truth:
                total += 1
                by_country[ctry][0] += 1
                if tid in merged_ids:
                    continue
                missed += 1
                by_country[ctry][1] += 1
                lr = lex_rank.get(tid)
                dr = dense_rank.get(tid)
                if (lr is not None and lr <= args.prod_lex) or (dr is not None and dr <= args.prod_dense):
                    cat = "pool_but_cut"
                elif lr is not None or dr is not None:
                    cat = "ranked_too_deep"
                else:
                    cat = "hard_miss"
                cats[cat] += 1
                if cat == "hard_miss" and len(examples) < args.examples:
                    p = pos_of.get(tid)
                    examples.append((entity_id, ctry, name, address, tid,
                                     cand_name[p] if p is not None else "?",
                                     cand_addr[p] if p is not None else "?"))

    print(f"\n[miss] true pairs={total:,}  missed@{args.prod_top}={missed:,}  "
          f"recall@{args.prod_top}={1 - missed / total:.4f}")
    print("[miss] breakdown of the missed pairs:")
    for cat in ("pool_but_cut", "ranked_too_deep", "hard_miss"):
        n = cats[cat]
        print(f"  {cat:<16}{n:>8,}  ({n / missed:.1%} of misses)" if missed else f"  {cat:<16}{n:>8,}")
    print("[miss] miss rate by country (country of the S1 entity):")
    for ctry in sorted(by_country):
        t, m = by_country[ctry]
        print(f"  {ctry:<10} true={t:>8,}  missed={m:>8,}  miss_rate={m / t:.1%}")

    if examples:
        print(f"\n[miss] example hard_miss pairs (S1 vs the true match it never retrieved):")
        for sid, ctry, sname, saddr, tid, tname, taddr in examples:
            print(f"  -- {ctry} | S1 {sid}: {sname!r} | {saddr!r}")
            print(f"     missed true {tid}: {tname!r} | {taddr!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
