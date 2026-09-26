"""Memory-bounded inference and submission-file creation."""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from blocking import candidate_ids, load_sources
from features import FEATURE_NAMES, extract_features_for_hits
from hybrid_blocking import (
    DEFAULT_DENSE_K,
    DEFAULT_ENCODER,
    DEFAULT_ENCODE_BATCH,
    DEFAULT_LEXICAL_K,
    DEFAULT_TOP_K,
    HybridIndex,
    iter_country_batches,
    retrieve_batch,
)
from train_model import _decision_predictions


def load_model(model_dir: str):
    model_path = os.path.join(model_dir, "matcher_model.joblib")
    meta_path = os.path.join(model_dir, "model_meta.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"No trained model found in '{model_dir}'. Run train_model.py first.")
    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("Feature schema differs from the saved model. Retrain before prediction.")
    return model, meta


def predict(
    test_dir: str, model_dir: str, output_dir: str, top_k: int | None = None,
    lexical_k: int | None = None, dense_k: int | None = None,
    encoder: str | None = None, device: str = "auto",
    encode_batch: int | None = None, dense_enabled: bool | None = None,
    first_threshold: float | None = None, additional_threshold: float | None = None,
    s1_limit: int | None = None, cand_limit: int | None = None,
    batch_size: int = 1_000,
) -> None:
    model, meta = load_model(model_dir)
    # Retrieval settings default to the ones the model was trained with so the
    # feature distributions at inference match training.
    top_k = top_k if top_k is not None else int(meta.get("top_k_blocking", DEFAULT_TOP_K))
    lexical_k = lexical_k if lexical_k is not None else int(meta.get("lexical_k", DEFAULT_LEXICAL_K))
    dense_k = dense_k if dense_k is not None else int(meta.get("dense_k", DEFAULT_DENSE_K))
    encoder = encoder if encoder is not None else str(meta.get("encoder") or DEFAULT_ENCODER)
    encode_batch = encode_batch if encode_batch is not None else int(meta.get("encode_batch", DEFAULT_ENCODE_BATCH))
    if dense_enabled is None:
        dense_enabled = bool(meta.get("dense_enabled", True))
    first_threshold = first_threshold if first_threshold is not None else float(meta["first_threshold"])
    additional_threshold = additional_threshold if additional_threshold is not None else float(meta["additional_threshold"])
    print(
        f"[predict] {meta['model_type']} | top_k={top_k} (lexical_k={lexical_k}, dense_k={dense_k}) | "
        f"encoder={encoder if dense_enabled else 'none'} | first={first_threshold:.2f} | "
        f"additional={additional_threshold:.2f} | validation F_0.5={meta.get('val_f05', 0):.4f}"
    )

    s1, s2, s3 = load_sources(test_dir, s1_limit=s1_limit)
    candidates = pd.concat([s2, s3], ignore_index=True)
    if cand_limit is not None:
        candidates = candidates.head(cand_limit).copy()
        print(f"[predict] cand_limit={cand_limit}: using {len(candidates):,} candidate rows "
              f"(SMOKE TEST ONLY - never submit these predictions)")
    entity_ids = candidates["entity_id"].to_numpy(copy=False)
    index = HybridIndex.build(
        candidates, test_dir, model_dir=model_dir, top_k=top_k,
        lexical_k=lexical_k, dense_k=dense_k, encoder=encoder, device=device,
        encode_batch=encode_batch, dense_enabled=dense_enabled, cand_limit=cand_limit,
    )
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    written = 0
    with open(matching_path, "w", encoding="utf-8", newline="") as matching, open(
        candidate_path, "w", encoding="utf-8", newline=""
    ) as candidate_file:
        matching.write("source1_entity_id\tmatched_entity_ids\n")
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        for batch in iter_country_batches(s1, batch_size=batch_size):
            hits_by_s1 = retrieve_batch(batch, index, top_k=top_k)
            X, source_ids, candidate_rows, _hits = extract_features_for_hits(batch, candidates, hits_by_s1)
            probabilities = model.predict_proba(X)[:, 1] if len(X) else np.empty(0, dtype=np.float32)
            predictions = _decision_predictions(
                source_ids, candidate_rows, probabilities, entity_ids, first_threshold, additional_threshold
            )
            for s1_id in batch["entity_id"]:
                candidate_file.write(f"{s1_id}\t{','.join(candidate_ids(hits_by_s1[s1_id], entity_ids))}\n")
                matching.write(f"{s1_id}\t{','.join(predictions.get(s1_id, []))}\n")
            written += len(batch)
            print(f"[predict] Wrote {written:,}/{len(s1):,} Source 1 rows")

    print(f"[predict] Wrote {matching_path}")
    print(f"[predict] Wrote {candidate_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict entity matches on the full test set")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--lexical-k", type=int, default=None, help="Default: the trained model's setting")
    parser.add_argument("--dense-k", type=int, default=None, help="Default: the trained model's setting")
    parser.add_argument("--encoder", default=None, help="Default: the trained model's setting")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch", type=int, default=None, help="Default: the trained model's setting")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--first-threshold", type=float, default=None)
    parser.add_argument("--additional-threshold", type=float, default=None)
    parser.add_argument("--s1-limit", type=int, default=None)
    parser.add_argument("--cand-limit", type=int, default=None,
                        help="SMOKE TEST ONLY: truncate S2/S3 candidates before indexing")
    parser.add_argument("--batch-size", type=int, default=1_000)
    args = parser.parse_args()
    predict(
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        lexical_k=args.lexical_k,
        dense_k=args.dense_k,
        encoder=args.encoder,
        device=args.device,
        encode_batch=args.encode_batch,
        dense_enabled=(False if args.no_dense else None),
        first_threshold=args.first_threshold,
        additional_threshold=args.additional_threshold,
        s1_limit=args.s1_limit,
        cand_limit=args.cand_limit,
        batch_size=args.batch_size,
    )
