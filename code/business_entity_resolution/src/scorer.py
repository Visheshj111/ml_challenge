import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def f05_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def compute_entity_f05(predicted: list, truth: set) -> float:
    predicted_set = set(predicted)
    if not truth and not predicted_set:
        return 1.0  # I correctly called it a singleton, full credit
    if not truth:
        return 0.0  # I made up a match where there was none — F0.5 penalises this hard
    if not predicted_set:
        precision = 0.0
        recall = 0.0
    else:
        tp = len(predicted_set & truth)
        precision = tp / len(predicted_set)
        recall = tp / len(truth)
    return f05_score(precision, recall)


def compute_validation_f05(
    predictions: dict,
    ground_truth: dict,
    s1_ids: list = None,
) -> float:
    if s1_ids is None:
        s1_ids = list(ground_truth.keys())

    scores = []
    for s1_id in s1_ids:
        pred = predictions.get(s1_id, [])
        truth = ground_truth.get(s1_id, set())
        scores.append(compute_entity_f05(pred, truth))

    return float(np.mean(scores)) if scores else 0.0


def score_submission(matching_path: str, gt_path: str) -> dict:
    pred_df = pd.read_csv(matching_path, sep="\t", dtype=str).fillna("")
    predictions = {
        s1_id: (matched.split(",") if matched.strip() else [])
        for s1_id, matched in zip(pred_df["source1_entity_id"], pred_df["matched_entity_ids"])
    }

    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    ground_truth = {
        s1_id: (set(matched.split(",")) if matched.strip() else set())
        for s1_id, matched in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"])
    }

    s1_ids = list(ground_truth.keys())
    entity_scores = []
    n_singletons = 0
    n_correct_singletons = 0

    for s1_id in s1_ids:
        pred = predictions.get(s1_id, [])
        truth = ground_truth.get(s1_id, set())
        score = compute_entity_f05(pred, truth)
        entity_scores.append(score)

        if not truth:
            n_singletons += 1
            if not pred:
                n_correct_singletons += 1

    macro_f05 = float(np.mean(entity_scores)) if entity_scores else 0.0

    # I compute micro stats separately — they're not the leaderboard metric but they help me debug
    # whether the problem is precision, recall, or both
    total_tp = 0; total_pred = 0; total_true = 0
    for s1_id in s1_ids:
        pred_set = set(predictions.get(s1_id, []))
        truth = ground_truth.get(s1_id, set())
        total_tp += len(pred_set & truth)
        total_pred += len(pred_set)
        total_true += len(truth)

    micro_precision = total_tp / total_pred if total_pred > 0 else 0.0
    micro_recall    = total_tp / total_true if total_true > 0 else 0.0
    micro_f05 = f05_score(micro_precision, micro_recall)

    return {
        "macro_f05": macro_f05,
        "micro_f05": micro_f05,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "n_entities": len(s1_ids),
        "n_singletons": n_singletons,
        "n_correct_singletons": n_correct_singletons,
        "singleton_accuracy": (n_correct_singletons / n_singletons
                               if n_singletons > 0 else None),
    }


def print_score_report(stats: dict):
    print("\n" + "="*50)
    print(f"  Macro F_0.5    : {stats['macro_f05']:.4f}  ← leaderboard metric")
    print(f"  Micro F_0.5    : {stats['micro_f05']:.4f}")
    print(f"  Micro Precision: {stats['micro_precision']:.4f}")
    print(f"  Micro Recall   : {stats['micro_recall']:.4f}")
    print(f"  Entities scored: {stats['n_entities']}")
    print(f"  True singletons: {stats['n_singletons']} "
          f"({100*stats['n_singletons']/max(stats['n_entities'],1):.1f}%)")
    if stats['singleton_accuracy'] is not None:
        print(f"  Singleton accuracy: {stats['singleton_accuracy']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score matching_results.tsv against ground truth")
    parser.add_argument("--matching",      required=True, help="Path to matching_results.tsv")
    parser.add_argument("--ground-truth",  required=True, help="Path to train_ground_truth.tsv")
    args = parser.parse_args()

    stats = score_submission(args.matching, args.ground_truth)
    print_score_report(stats)
