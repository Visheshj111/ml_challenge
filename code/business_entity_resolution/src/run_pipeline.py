"""Run training, full test inference, and official submission validation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from hybrid_blocking import (
    DEFAULT_DENSE_K,
    DEFAULT_ENCODER,
    DEFAULT_ENCODE_BATCH,
    DEFAULT_LEXICAL_K,
    DEFAULT_TOP_K,
)
from predict import predict
from train_model import train


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end business entity-resolution pipeline")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--model-dir", default="models/hybrid_v1")
    parser.add_argument("--output-dir", default="output/hybrid_v1")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--lexical-k", type=int, default=None,
                        help=f"Lexical pool size (default: {DEFAULT_LEXICAL_K})")
    parser.add_argument("--dense-k", type=int, default=None,
                        help=f"Dense pool size (default: {DEFAULT_DENSE_K})")
    parser.add_argument("--encoder", default=None,
                        help=f"Dense encoder model name (default: {DEFAULT_ENCODER})")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch", type=int, default=None,
                        help=f"Encoder batch size (default: {DEFAULT_ENCODE_BATCH})")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--train-s1-sample", type=int, default=50_000)
    parser.add_argument("--validation-s1-sample", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--s1-limit", type=int, default=None, help="Smoke-test only; validator will reject limited predictions.")
    parser.add_argument("--cand-limit", type=int, default=None,
                        help="Smoke-test only; truncates S2/S3 candidates before indexing.")
    parser.add_argument("--predict-batch-size", type=int, default=5_000,
                        help="S1 rows per retrieval batch during prediction; larger batches "
                             "re-stream each country segment less often but use more host RAM")
    args = parser.parse_args()

    lexical_k = args.lexical_k if args.lexical_k is not None else DEFAULT_LEXICAL_K
    dense_k = args.dense_k if args.dense_k is not None else DEFAULT_DENSE_K
    encoder = args.encoder if args.encoder is not None else DEFAULT_ENCODER
    encode_batch = args.encode_batch if args.encode_batch is not None else DEFAULT_ENCODE_BATCH

    print("=" * 60)
    print("Amazon ML Challenge - Entity Resolution (hybrid retrieval)")
    print("=" * 60)
    if args.skip_train:
        print("[pipeline] Reusing the saved model.")
    else:
        print("[pipeline] Step 1/3: training")
        train(
            train_dir=args.train_dir,
            model_dir=args.model_dir,
            top_k=args.top_k,
            lexical_k=lexical_k,
            dense_k=dense_k,
            encoder=encoder,
            device=args.device,
            encode_batch=encode_batch,
            dense_enabled=not args.no_dense,
            cand_limit=args.cand_limit,
            train_s1_sample=args.train_s1_sample,
            validation_s1_sample=args.validation_s1_sample,
            random_seed=args.seed,
            s1_limit=args.s1_limit,
        )

    print("[pipeline] Step 2/3: full test prediction")
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
        s1_limit=args.s1_limit,
        cand_limit=args.cand_limit,
        batch_size=args.predict_batch_size,
    )

    print("[pipeline] Step 3/3: official format validation")
    validator = ROOT / "utils" / "validate_submission.py"
    matching = Path(args.output_dir) / "matching_results.tsv"
    candidates = Path(args.output_dir) / "candidate_pairs.tsv"
    result = subprocess.run(
        [sys.executable, str(validator), "--matching", str(matching), "--candidate", str(candidates), "--test-dir", args.test_dir],
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise SystemExit("[pipeline] Submission validation failed.")
    print(f"[pipeline] PASS. Upload {matching} to the leaderboard.")


if __name__ == "__main__":
    main()
