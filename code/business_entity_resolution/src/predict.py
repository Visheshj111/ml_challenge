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
from rerank import build_candidate_texts, build_s1_texts, load_cross_encoder, rerank_batch_probabilities
from tune_decisions import _build_groups, _predict_with_rules


def _load_decision_rules(path: str) -> dict[str, tuple[float, float, int | None]]:
    """Load country -> (first, additional, cap) rules from a calibration JSON.

    Accepts either the full tune_decisions/evaluate_rerank output (with a
    'recommended' block) or a bare rules mapping. Country keys are lowercased
    to match the lookup predict.py performs on the S1 country values.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("recommended"), dict):
        raw = payload["recommended"].get("rules", {})
    else:
        raw = payload
    rules: dict[str, tuple[float, float, int | None]] = {}
    for country, rule in raw.items():
        key = "*" if country == "*" else str(country).strip().lower()
        rules[key] = (float(rule["first"]), float(rule["additional"]), rule.get("cap"))
    if "*" not in rules:
        raise ValueError(f"Decision rules in '{path}' have no '*' fallback rule.")
    return rules


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
    max_matches: int | None = None, decision_rules_path: str | None = None,
    cross_encoder: str | None = None, ce_top_n: int = 10, ce_batch_size: int = 256,
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
    if decision_rules_path:
        rules = _load_decision_rules(decision_rules_path)
    else:
        rules = {"*": (first_threshold, additional_threshold, max_matches)}
    print(
        f"[predict] {meta['model_type']} | top_k={top_k} (lexical_k={lexical_k}, dense_k={dense_k}) | "
        f"encoder={encoder if dense_enabled else 'none'} | validation F_0.5={meta.get('val_f05', 0):.4f}"
    )
    rules_desc = ", ".join(
        f"{c}: first={r[0]:.3f} add={r[1]:.3f} cap={r[2]}" for c, r in sorted(rules.items())
    )
    print(f"[predict] Decision rules -> {rules_desc}")
    ce = load_cross_encoder(cross_encoder, device=device) if cross_encoder else None
    if ce is not None:
        print(f"[predict] Cross-encoder rerank active: top_n={ce_top_n} per entity.")

    s1, s2, s3 = load_sources(test_dir, s1_limit=s1_limit)
    candidates = pd.concat([s2, s3], ignore_index=True)
    if cand_limit is not None:
        candidates = candidates.head(cand_limit).copy()
        print(f"[predict] cand_limit={cand_limit}: using {len(candidates):,} candidate rows "
              f"(SMOKE TEST ONLY - never submit these predictions)")
    entity_ids = candidates["entity_id"].to_numpy(copy=False)
    cand_texts = build_candidate_texts(candidates) if ce is not None else None
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
            if ce is not None:
                probabilities = rerank_batch_probabilities(
                    build_s1_texts(batch), cand_texts, source_ids, candidate_rows,
                    probabilities, ce, top_n=ce_top_n, ce_batch_size=ce_batch_size,
                )
            groups = _build_groups(source_ids, candidate_rows, probabilities, entity_ids)
            s1_countries = {
                str(entity_id): str(country).strip().lower()
                for entity_id, country in batch[["entity_id", "country"]].itertuples(index=False, name=None)
            }
            predictions = _predict_with_rules(groups, s1_countries, rules)
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
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Cap on accepted matches per Source 1 entity (recommended: 5).")
    parser.add_argument("--decision-rules", default=None,
                        help="Calibration JSON (decision_calibration.json or rerank_calibration.json); "
                             "overrides --first/--additional-threshold and --max-matches.")
    parser.add_argument("--cross-encoder", default=None,
                        help="Cross-encoder model dir or HF name; reranks the top-N pairs per entity.")
    parser.add_argument("--ce-top-n", type=int, default=10)
    parser.add_argument("--ce-batch-size", type=int, default=256)
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
        max_matches=args.max_matches,
        decision_rules_path=args.decision_rules,
        cross_encoder=args.cross_encoder,
        ce_top_n=args.ce_top_n,
        ce_batch_size=args.ce_batch_size,
        s1_limit=args.s1_limit,
        cand_limit=args.cand_limit,
        batch_size=args.batch_size,
    )
