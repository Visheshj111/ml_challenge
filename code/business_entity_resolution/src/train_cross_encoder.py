

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from datasets import Dataset

from blocking import load_sources
from hybrid_blocking import (
    DEFAULT_DENSE_K,
    DEFAULT_ENCODE_BATCH,
    DEFAULT_ENCODER,
    DEFAULT_LEXICAL_K,
    DEFAULT_TOP_K,
    HybridIndex,
    retrieve_batch as hybrid_retrieve_batch,
    iter_country_batches,
)


def load_ground_truth(path: str) -> dict[str, set[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        s1_id: set(matches.split(",")) if matches else set()
        for s1_id, matches in frame[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False, name=None)
    }


def format_text(name, address) -> str:
    n = str(name).strip() if pd.notna(name) else ""
    a = str(address).strip() if pd.notna(address) else ""
    return f"{n} - {a}".strip()


def build_training_pairs(
    s1_subset: pd.DataFrame,
    candidates: pd.DataFrame,
    index: HybridIndex,
    ground_truth: dict[str, set[str]],
    candidate_ids: np.ndarray,
    top_k: int,
) -> tuple[list[str], list[str], list[float]]:
    """Return parallel lists: (query_texts, candidate_texts, labels)."""
    queries: list[str] = []
    docs: list[str] = []
    labels: list[float] = []
    processed = 0

    for batch in iter_country_batches(s1_subset, batch_size=1_000):
        hits_by_s1 = hybrid_retrieve_batch(batch, index, top_k=top_k)

        for s1_entity_id, s1_name, s1_addr, _country in batch[
            ["entity_id", "business_name", "business_address", "country"]
        ].itertuples(index=False, name=None):

            s1_text = format_text(s1_name, s1_addr)
            truth = ground_truth.get(s1_entity_id, set())
            hits = hits_by_s1.get(s1_entity_id, [])

            for hit in hits:
                cand_id = candidate_ids[hit.row]
                cand_row = candidates.iloc[hit.row]
                cand_text = format_text(cand_row["business_name"], cand_row["business_address"])

                queries.append(s1_text)
                docs.append(cand_text)
                labels.append(1.0 if cand_id in truth else 0.0)

        processed += len(batch)
        if processed % 10_000 == 0 or processed == len(s1_subset):
            pos = sum(labels[-len(batch) * top_k:])
            print(f"[cross-encoder] {processed}/{len(s1_subset)} S1 rows processed | pairs so far: {len(labels):,}")

    return queries, docs, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--model-dir", default="models/cross_encoder")
    parser.add_argument("--hybrid-model-dir", default="models/hybrid_v1",
                        help="Model dir that holds the dense embedding cache (avoids re-encoding).")
    parser.add_argument("--s1-limit", type=int, default=None,
                        help="Limit S1 rows for quick smoke tests.")
    parser.add_argument("--cand-limit", type=int, default=None,
                        help="Limit candidate pool for quick smoke tests.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2",
                        help="HuggingFace cross-encoder checkpoint to fine-tune.")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("[cross-encoder] Loading training data...")
    s1_train, s2, s3 = load_sources(args.train_dir, s1_limit=args.s1_limit)
    candidates = pd.concat([s2, s3], ignore_index=True)
    cand_ids = candidates["entity_id"].to_numpy(copy=False)
    ground_truth = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))

    print("[cross-encoder] Building Hybrid index (reuses cache if warm)...")
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

    print("[cross-encoder] Generating training pairs from hybrid retrieval...")
    queries, docs, labels = build_training_pairs(
        s1_train, candidates, index, ground_truth, cand_ids, top_k=args.top_k
    )
    print(f"[cross-encoder] Total pairs: {len(labels):,} | positives: {int(sum(labels)):,} ({100*sum(labels)/max(len(labels),1):.2f}%)")

    # Shuffle in-place
    combined = list(zip(queries, docs, labels))
    random.shuffle(combined)
    queries, docs, labels = zip(*combined)

    # Build HuggingFace Dataset compatible with sentence-transformers v3+
    train_dataset = Dataset.from_dict({
        "sentence1": list(queries),
        "sentence2": list(docs),
        "label": list(labels),
    })

    print(f"[cross-encoder] Loading base model: {args.base_model} ...")
    model = CrossEncoder(args.base_model, num_labels=1, device=args.device if args.device != "auto" else None)

    training_args = CrossEncoderTrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    print("[cross-encoder] Starting training...")
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    model.save_pretrained(args.model_dir)
    print(f"[cross-encoder] Model saved to: {args.model_dir}")


if __name__ == "__main__":
    main()
