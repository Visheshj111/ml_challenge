"""Training for the retrieve-and-rerank entity-resolution model."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from blocking import load_sources
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
from features import FEATURE_NAMES, extract_features_for_hits
from scorer import compute_validation_f05

try:
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False

try:
    import xgboost as xgb
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False


def load_ground_truth(path: str) -> dict[str, set[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        s1_id: set(matches.split(",")) if matches else set()
        for s1_id, matches in frame[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False, name=None)
    }


def _select_hard_negatives(labels: np.ndarray, source_ids: list[str]) -> np.ndarray:
    """Keep all positives and the highest-ranked retrieved negatives per S1."""
    keep: list[int] = []
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and source_ids[end] == source_ids[start]:
            end += 1
        group = np.arange(start, end)
        positive = group[labels[group] == 1]
        negative = group[labels[group] == 0]
        keep.extend(positive.tolist())
        # Candidate hits are emitted in retrieval-rank order. Singleton examples
        # retain negatives too, teaching the matcher that a plausible candidate
        # does not necessarily imply a real match.
        keep_negatives = min(len(negative), max(3, min(10, 5 * len(positive))))
        keep.extend(negative[:keep_negatives].tolist())
        start = end
    return np.asarray(keep, dtype=np.int64)


def _candidate_recall(
    hits_by_s1,
    ground_truth: dict[str, set[str]],
    candidate_ids: np.ndarray,
    ks: tuple[int, ...] = (25, 50, 100),
) -> tuple[dict[int, float], int, int]:
    found = {k: 0 for k in ks}
    total = 0
    for s1_id, hits in hits_by_s1.items():
        truth = ground_truth.get(s1_id, set())
        total += len(truth)
        if not truth:
            continue
        ids = [str(candidate_ids[hit.row]) for hit in hits]
        for k in ks:
            found[k] += sum(match in ids[:k] for match in truth)
    return ({k: found[k] / total if total else 0.0 for k in ks}, sum(found.values()), total)


def _decision_predictions(
    source_ids: list[str], candidate_rows: list[int], probabilities: np.ndarray,
    candidate_ids: np.ndarray, first_threshold: float, additional_threshold: float,
) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for s1_id, row, probability in zip(source_ids, candidate_rows, probabilities):
        grouped[s1_id].append((float(probability), row))
    predictions: dict[str, list[str]] = {}
    for s1_id, pairs in grouped.items():
        pairs.sort(reverse=True)
        if not pairs or pairs[0][0] < first_threshold:
            predictions[s1_id] = []
            continue
        accepted = [str(candidate_ids[row]) for score, row in pairs if score >= additional_threshold]
        if str(candidate_ids[pairs[0][1]]) not in accepted:
            accepted.insert(0, str(candidate_ids[pairs[0][1]]))
        predictions[s1_id] = accepted
    return predictions


def optimise_thresholds(
    source_ids: list[str], candidate_rows: list[int], probabilities: np.ndarray,
    candidate_ids: np.ndarray, ground_truth: dict[str, set[str]], validation_ids: list[str],
) -> tuple[float, float, float]:
    first_grid = np.arange(0.45, 0.96, 0.05)
    best = (0.50, 0.70, -1.0)
    for first in first_grid:
        for additional in np.arange(first, 0.991, 0.05):
            predictions = _decision_predictions(
                source_ids, candidate_rows, probabilities, candidate_ids, float(first), float(additional)
            )
            score = compute_validation_f05(predictions, ground_truth, validation_ids)
            if score > best[2]:
                best = (float(first), float(additional), float(score))
    return best


def _build_matrix(
    s1_subset: pd.DataFrame, candidates: pd.DataFrame, index: HybridIndex,
    ground_truth: dict[str, set[str]], candidate_ids: np.ndarray, top_k: int,
    batch_size: int, select_hard_negatives: bool, ks: tuple[int, ...] = (25, 50),
) -> tuple[np.ndarray, np.ndarray, list[str], list[int], dict[int, float]]:
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    all_source_ids: list[str] = []
    all_candidate_rows: list[int] = []
    recall_found = {k: 0.0 for k in ks}
    recall_total = 0

    prepared_rows = 0
    for batch in iter_country_batches(s1_subset, batch_size=batch_size):
        hits_by_s1 = retrieve_batch(batch, index, top_k=top_k)
        recall, _unused, total = _candidate_recall(hits_by_s1, ground_truth, candidate_ids, ks=ks)
        for k, value in recall.items():
            recall_found[k] += value * total
        recall_total += total

        X_batch, source_ids, candidate_rows, _hits = extract_features_for_hits(batch, candidates, hits_by_s1)
        prepared_rows += len(batch)
        if len(X_batch) == 0:
            print(f"[train] Prepared {prepared_rows:,}/{len(s1_subset):,} S1 rows (no candidates)")
            continue
        labels = np.fromiter(
            (int(str(candidate_ids[row]) in ground_truth.get(s1_id, set())) for s1_id, row in zip(source_ids, candidate_rows)),
            dtype=np.int8,
            count=len(source_ids),
        )
        if select_hard_negatives:
            keep = _select_hard_negatives(labels, source_ids)
            X_batch = X_batch[keep]
            labels = labels[keep]
            source_ids = [source_ids[i] for i in keep]
            candidate_rows = [candidate_rows[i] for i in keep]
        feature_chunks.append(X_batch)
        label_chunks.append(labels)
        all_source_ids.extend(source_ids)
        all_candidate_rows.extend(candidate_rows)
        print(f"[train] Prepared {prepared_rows:,}/{len(s1_subset):,} S1 rows")

    if not feature_chunks:
        raise RuntimeError("Blocking produced no candidate pairs. Check the source files and retrieval settings.")
    recall_result = {k: recall_found[k] / recall_total if recall_total else 0.0 for k in recall_found}
    return np.vstack(feature_chunks), np.concatenate(label_chunks), all_source_ids, all_candidate_rows, recall_result


def train(
    train_dir: str, model_dir: str = "models", top_k: int = DEFAULT_TOP_K,
    lexical_k: int = DEFAULT_LEXICAL_K, dense_k: int = DEFAULT_DENSE_K,
    encoder: str = DEFAULT_ENCODER, device: str = "auto",
    encode_batch: int = DEFAULT_ENCODE_BATCH, dense_enabled: bool = True,
    cand_limit: int | None = None, train_s1_sample: int = 50_000,
    validation_s1_sample: int = 10_000, random_seed: int = 42,
    s1_limit: int | None = None,
) -> tuple[object, dict]:
    if not HAVE_LIGHTGBM and not HAVE_XGBOOST:
        raise RuntimeError("Install lightgbm or xgboost before training.")
    os.makedirs(model_dir, exist_ok=True)
    s1, s2, s3 = load_sources(train_dir, s1_limit=s1_limit)
    candidates = pd.concat([s2, s3], ignore_index=True)
    if cand_limit is not None:
        candidates = candidates.head(cand_limit).copy()
        print(f"[train] cand_limit={cand_limit}: using {len(candidates):,} candidate rows "
              f"(SMOKE TEST ONLY - never submit a model trained this way)")
    candidate_ids = candidates["entity_id"].to_numpy(copy=False)
    ground_truth = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(s1))
    validation_count = min(validation_s1_sample, max(1, len(s1) // 5))
    train_count = min(train_s1_sample, max(1, len(s1) - validation_count))
    validation_rows = order[:validation_count]
    training_rows = order[validation_count:validation_count + train_count]
    s1_train = s1.iloc[training_rows].reset_index(drop=True)
    s1_validation = s1.iloc[validation_rows].reset_index(drop=True)
    validation_ids = s1_validation["entity_id"].tolist()
    print(f"[train] Training S1 sample: {len(s1_train):,}; validation S1 sample: {len(s1_validation):,}")

    index = HybridIndex.build(
        candidates, train_dir, model_dir=model_dir, top_k=top_k,
        lexical_k=lexical_k, dense_k=dense_k, encoder=encoder, device=device,
        encode_batch=encode_batch, dense_enabled=dense_enabled, cand_limit=cand_limit,
    )
    print("[train] Generating production-like training candidates and hard negatives ...")
    X_train, y_train, _train_sids, _train_rows, train_recall = _build_matrix(
        s1_train, candidates, index, ground_truth, candidate_ids, top_k,
        batch_size=1_000, select_hard_negatives=True, ks=(25, 50, top_k),
    )
    print(f"[train] Candidate recall: " + ", ".join(f"@{k}={v:.4f}" for k, v in train_recall.items()))
    print(f"[train] Training pairs: {len(X_train):,}; positives: {int(y_train.sum()):,}")

    print("[train] Generating held-out validation candidates ...")
    X_val, y_val, val_sids, val_rows, val_recall = _build_matrix(
        s1_validation, candidates, index, ground_truth, candidate_ids, top_k,
        batch_size=1_000, select_hard_negatives=False, ks=(25, 50, top_k),
    )
    print(f"[train] Validation candidate recall: " + ", ".join(f"@{k}={v:.4f}" for k, v in val_recall.items()))

    if HAVE_LIGHTGBM:
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=800, learning_rate=0.05, num_leaves=63,
            max_depth=-1, min_child_samples=20, subsample=0.8, colsample_bytree=0.9,
            reg_lambda=1.0, n_jobs=-1, random_state=random_seed, verbose=-1,
        )
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        model_type = "lightgbm"
    else:
        model = xgb.XGBClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=7, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1,
            random_state=random_seed, early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model_type = "xgboost"

    probabilities = model.predict_proba(X_val)[:, 1]
    first_threshold, additional_threshold, validation_f05 = optimise_thresholds(
        val_sids, val_rows, probabilities, candidate_ids, ground_truth, validation_ids
    )
    print(
        f"[train] Best decision thresholds: first={first_threshold:.2f}, "
        f"additional={additional_threshold:.2f}, validation macro F_0.5={validation_f05:.4f}"
    )

    model_path = os.path.join(model_dir, "matcher_model.joblib")
    meta_path = os.path.join(model_dir, "model_meta.json")
    joblib.dump(model, model_path)
    meta = {
        "model_type": model_type,
        "feature_names": FEATURE_NAMES,
        "top_k_blocking": top_k,
        "lexical_k": lexical_k,
        "dense_k": dense_k,
        "encoder": index.dense.encoder if index.dense is not None else None,
        "encode_batch": encode_batch,
        "dense_enabled": index.dense is not None,
        "first_threshold": first_threshold,
        "additional_threshold": additional_threshold,
        "val_f05": validation_f05,
        "train_candidate_recall": train_recall,
        "validation_candidate_recall": val_recall,
        "train_s1_sample": len(s1_train),
        "validation_s1_sample": len(s1_validation),
        "random_seed": random_seed,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[train] Saved model to {model_path}")
    return model, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the entity-resolution matcher")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--lexical-k", type=int, default=DEFAULT_LEXICAL_K)
    parser.add_argument("--dense-k", type=int, default=DEFAULT_DENSE_K)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch", type=int, default=DEFAULT_ENCODE_BATCH)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--cand-limit", type=int, default=None,
                        help="SMOKE TEST ONLY: truncate S2/S3 candidates before indexing")
    parser.add_argument("--train-s1-sample", type=int, default=50_000)
    parser.add_argument("--validation-s1-sample", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--s1-limit", type=int, default=None,
                        help="SMOKE TEST ONLY: limit S1 rows loaded")
    args = parser.parse_args()
    train(
        train_dir=args.train_dir,
        model_dir=args.model_dir,
        top_k=args.top_k,
        lexical_k=args.lexical_k,
        dense_k=args.dense_k,
        encoder=args.encoder,
        device=args.device,
        encode_batch=args.encode_batch,
        dense_enabled=not args.no_dense,
        cand_limit=args.cand_limit,
        train_s1_sample=args.train_s1_sample,
        validation_s1_sample=args.validation_s1_sample,
        random_seed=args.seed,
        s1_limit=args.s1_limit,
    )
