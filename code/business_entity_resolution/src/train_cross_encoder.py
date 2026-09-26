"""Fine-tune a multilingual cross-encoder for business-entity matching.

Training pairs come from the same hybrid retrieval used at inference. For
every sampled Source 1 row each retrieved candidate is labelled from the
train ground truth; negatives are subsampled per entity (all positives are
kept, and the retrieval-hardest negative is always kept) so one epoch over
~100k entities is a few million pairs.

The S1 sample starts at order[85_000] of the same seed-42 permutation
train_model.py uses, so it is disjoint from the LightGBM validation rows
(order[:10000]), training rows (order[10000:60000]), and the fresh
calibration slice (order[60000:85000]) that evaluate_rerank.py scores.

--pairs-out / --pairs-in let pair generation and training run on different
machines: generate locally against a warm embedding cache, then train on a
cloud GPU without re-encoding 10M candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sklearn.metrics import roc_auc_score

from blocking import load_sources
from hybrid_blocking import (
    DEFAULT_DENSE_K,
    DEFAULT_ENCODE_BATCH,
    DEFAULT_ENCODER,
    DEFAULT_LEXICAL_K,
    DEFAULT_TOP_K,
    HybridIndex,
    iter_country_batches,
    retrieve_batch as hybrid_retrieve_batch,
)
from rerank import CE_MAX_LENGTH, build_candidate_texts, build_s1_texts

SEED = 42
S1_SKIP = 85_000  # 10k validation + 50k LGBM train + 25k calibration slice
DEFAULT_BASE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def load_ground_truth(path: str) -> dict[str, set[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        s1_id: set(matches.split(",")) if matches else set()
        for s1_id, matches in frame[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False, name=None)
    }


def generate_pairs(
    s1_subset: pd.DataFrame,
    candidates: pd.DataFrame,
    index: HybridIndex,
    ground_truth: dict[str, set[str]],
    cand_ids: np.ndarray,
    cand_texts: np.ndarray,
    top_k: int,
    neg_ratio: float,
    min_neg: int,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[float]]:
    queries: list[str] = []
    docs: list[str] = []
    labels: list[float] = []
    processed = 0
    for batch in iter_country_batches(s1_subset, batch_size=1_000):
        s1_texts = build_s1_texts(batch)
        hits_by_s1 = hybrid_retrieve_batch(batch, index, top_k=top_k)
        for s1_entity_id, _name, _addr, _country in batch[
            ["entity_id", "business_name", "business_address", "country"]
        ].itertuples(index=False, name=None):
            truth = ground_truth.get(s1_entity_id, set())
            pos_rows: list[int] = []
            neg_rows: list[int] = []
            for hit in hits_by_s1.get(s1_entity_id, []):
                (pos_rows if str(cand_ids[hit.row]) in truth else neg_rows).append(int(hit.row))
            for row in pos_rows:
                queries.append(s1_texts[s1_entity_id])
                docs.append(cand_texts[row])
                labels.append(1.0)
            if neg_rows:
                keep = min(len(neg_rows), max(min_neg, math.ceil(len(pos_rows) * neg_ratio)))
                chosen = [neg_rows[0]]
                if keep > 1 and len(neg_rows) > 1:
                    pool = neg_rows[1:]
                    chosen.extend(int(r) for r in rng.choice(pool, size=min(keep - 1, len(pool)), replace=False))
                for row in chosen:
                    queries.append(s1_texts[s1_entity_id])
                    docs.append(cand_texts[row])
                    labels.append(0.0)
        processed += len(batch)
        if processed % 10_000 < len(batch) or processed == len(s1_subset):
            print(f"[cross-encoder] {processed:,}/{len(s1_subset):,} S1 rows | "
                  f"pairs so far: {len(labels):,} | positives: {int(sum(labels)):,}")
    return queries, docs, labels


def save_pairs(path: str, queries, docs, labels) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("query\tdoc\tlabel\n")
        for q, d, l in zip(queries, docs, labels):
            handle.write(f"{q}\t{d}\t{int(l)}\n")
    print(f"[cross-encoder] Wrote {len(labels):,} pairs to {path}")


def load_pairs(path: str) -> tuple[list[str], list[str], list[float]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return (
        frame["query"].tolist(),
        frame["doc"].tolist(),
        frame["label"].astype(float).tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--model-dir", default="models/cross_encoder")
    parser.add_argument("--hybrid-model-dir", default="models/hybrid_v1",
                        help="Model dir holding the dense embedding cache (avoids re-encoding).")
    parser.add_argument("--s1-limit", type=int, default=100_000)
    parser.add_argument("--cand-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="HuggingFace cross-encoder checkpoint to fine-tune.")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER,
                        help="Bi-encoder used for retrieval when generating pairs.")
    parser.add_argument("--neg-ratio", type=float, default=4.0,
                        help="Negatives kept per positive (all positives are always kept).")
    parser.add_argument("--min-neg", type=int, default=3,
                        help="Minimum negatives kept per entity, even singletons.")
    parser.add_argument("--eval-frac", type=float, default=0.02,
                        help="Held-out pair fraction for the post-training AUC.")
    parser.add_argument("--pairs-out", default=None,
                        help="Save generated pairs to this TSV (upload-friendly).")
    parser.add_argument("--pairs-in", default=None,
                        help="Skip generation and train from this TSV instead.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    if args.pairs_in:
        print(f"[cross-encoder] Loading pairs from {args.pairs_in} ...")
        queries, docs, labels = load_pairs(args.pairs_in)
    else:
        print("[cross-encoder] Loading training data ...")
        s1, s2, s3 = load_sources(args.train_dir)
        candidates = pd.concat([s2, s3], ignore_index=True)
        if args.cand_limit is not None:
            candidates = candidates.head(args.cand_limit).copy()
        cand_ids = candidates["entity_id"].to_numpy(copy=False)
        ground_truth = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))

        order = np.random.default_rng(SEED).permutation(len(s1))
        take = min(args.s1_limit, max(len(s1) - S1_SKIP, 0))
        s1_subset = s1.iloc[order[S1_SKIP:S1_SKIP + take]].reset_index(drop=True)
        print(f"[cross-encoder] S1 sample: {len(s1_subset):,} rows from order[{S1_SKIP}:] "
              f"(disjoint from LGBM train/val and the calibration slice)")

        print("[cross-encoder] Building hybrid index (reuses a warm cache if present) ...")
        index = HybridIndex.build(
            candidates, args.train_dir,
            model_dir=args.hybrid_model_dir,
            top_k=args.top_k,
            lexical_k=DEFAULT_LEXICAL_K,
            dense_k=DEFAULT_DENSE_K,
            encoder=args.encoder,
            device=args.device,
            encode_batch=DEFAULT_ENCODE_BATCH,
            dense_enabled=True,
            cand_limit=args.cand_limit,
        )
        cand_texts = build_candidate_texts(candidates)
        print("[cross-encoder] Generating training pairs from hybrid retrieval ...")
        queries, docs, labels = generate_pairs(
            s1_subset, candidates, index, ground_truth, cand_ids, cand_texts,
            top_k=args.top_k, neg_ratio=args.neg_ratio, min_neg=args.min_neg, rng=rng,
        )
        if args.pairs_out:
            save_pairs(args.pairs_out, queries, docs, labels)

    n_pos = int(sum(labels))
    print(f"[cross-encoder] Total pairs: {len(labels):,} | positives: {n_pos:,} "
          f"({100 * n_pos / max(len(labels), 1):.2f}%)")

    combined = list(zip(queries, docs, labels))
    rng.shuffle(combined)
    queries, docs, labels = (list(x) for x in zip(*combined))
    n_eval = max(1, int(len(labels) * args.eval_frac))
    eval_queries, eval_docs, eval_labels = queries[:n_eval], docs[:n_eval], labels[:n_eval]
    train_dataset = Dataset.from_dict({
        "sentence1": queries[n_eval:],
        "sentence2": docs[n_eval:],
        "label": labels[n_eval:],
    })

    print(f"[cross-encoder] Loading base model: {args.base_model} ...")
    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=CE_MAX_LENGTH,
        device=(None if args.device == "auto" else args.device),
    )

    training_args = CrossEncoderTrainingArguments(
        output_dir=os.path.join(args.model_dir, "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        save_steps=2000,
        save_total_limit=1,
        load_best_model_at_end=False,
    )
    trainer = CrossEncoderTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(args.model_dir)

    eval_pairs = list(zip(eval_queries, eval_docs))
    eval_scores = np.asarray(
        model.predict(eval_pairs, batch_size=args.batch_size, show_progress_bar=False),
        dtype=np.float64,
    ).reshape(-1)
    auc = float(roc_auc_score(eval_labels, eval_scores)) if len(set(eval_labels)) > 1 else float("nan")
    print(f"[cross-encoder] Held-out pair AUC: {auc:.4f} on {len(eval_labels):,} pairs")

    meta = {
        "base_model": args.base_model,
        "pairs": len(labels),
        "positives": n_pos,
        "neg_ratio": args.neg_ratio,
        "min_neg": args.min_neg,
        "s1_limit": args.s1_limit,
        "s1_skip": S1_SKIP,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": CE_MAX_LENGTH,
        "heldout_auc": auc,
        "pairs_in": args.pairs_in,
    }
    with open(os.path.join(args.model_dir, "cross_encoder_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[cross-encoder] Model saved to: {args.model_dir}")


if __name__ == "__main__":
    main()
