# Business Entity Resolution Pipeline

This pipeline links every Source 1 business to zero, one, or many matching Source 2 and Source 3 businesses. It is designed for the challenge's macro F 0.5 metric, which makes incorrect matches to true singletons especially costly.

## Method

1. Prepare several reusable representations of each name and address: raw cleaned text, normalized text, legal-suffix-free core name, compact name, initials, significant tokens, and address numbers.
2. Build bounded, ranked retrieval indexes over Source 2 and Source 3. Candidate passes include exact normalized name, core name, compact name, rare-token anchors, numeric-address anchors, name-prefix anchors, and initials.
3. Score only retrieved candidate pairs with raw and normalized fuzzy similarities, name and address token evidence, numeric agreement and conflict, country evidence, and retrieval-rank features.
4. Train LightGBM or XGBoost using positives plus hard negatives from the same retrieval stage used at inference.
5. Select a first-match threshold and an additional-match threshold using held-out Source 1 entities and macro F 0.5.

The retrieval stage keeps a ranked candidate set. It does not build a dense Source 1 by Source 2/3 similarity matrix.

## Setup

From the repository root:

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
```

## First training run

Use the default sample sizes first. They provide a production-like candidate distribution while keeping model-training memory bounded on a 16 GB laptop:

```powershell
python code/business_entity_resolution/src/train_model.py --train-dir dataset/train --model-dir models --top-k 100 --train-s1-sample 50000 --validation-s1-sample 10000
```

Read the reported candidate Recall at 25, 50, and 100 and the held-out macro F 0.5 before scaling training higher.

## Full prediction and validation

After a model is saved, generate a complete leaderboard file:

```powershell
python code/business_entity_resolution/src/predict.py --test-dir dataset/test --model-dir models --output-dir output --top-k 100
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Only upload `output/matching_results.tsv` after the validator prints `PASS`.

## Full end-to-end run

This command trains, predicts, and validates. It uses the same sampled training size as above; increase `--train-s1-sample` only after reviewing the first run's validation metrics and available memory.

```powershell
python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test --model-dir models --output-dir output --top-k 100 --train-s1-sample 50000 --validation-s1-sample 10000
```

## Important constraints

- All input and output files are tab-separated.
- France is handled as an open country string and is never one-hot encoded as a fixed training category.
- Every Source 1 entity is written exactly once during full test prediction.
- All predicted IDs come from the retrieved candidate set, and the official validator is run before submission.
