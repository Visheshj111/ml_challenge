"""Cross-encoder reranking shared by prediction and slice evaluation.

The same text recipe must be used at cross-encoder training, slice
evaluation, and final prediction time, so it lives here exactly once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CE_MAX_LENGTH = 128


def cross_encoder_text(name, address) -> str:
    """One record flattened into a single cross-encoder sentence."""
    n = "" if pd.isna(name) else str(name).strip()
    a = "" if pd.isna(address) else str(address).strip()
    return f"{n} - {a}".strip()


def build_candidate_texts(candidates: pd.DataFrame) -> np.ndarray:
    """Precomputed candidate sentences indexed by candidate row."""
    texts = [
        cross_encoder_text(name, address)
        for name, address in zip(candidates["business_name"], candidates["business_address"])
    ]
    return np.array(texts, dtype=object)


def build_s1_texts(s1_batch: pd.DataFrame) -> dict[str, str]:
    return {
        str(entity_id): cross_encoder_text(name, address)
        for entity_id, name, address in s1_batch[
            ["entity_id", "business_name", "business_address"]
        ].itertuples(index=False, name=None)
    }


def load_cross_encoder(path_or_name: str, device: str = "auto"):
    """Load a fine-tuned model directory or a HuggingFace checkpoint."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        path_or_name,
        max_length=CE_MAX_LENGTH,
        device=(None if device == "auto" else device),
    )
    print(f"[rerank] Cross-encoder '{path_or_name}' ready (max_length={CE_MAX_LENGTH}).")
    return model


def rerank_batch_probabilities(
    s1_texts: dict[str, str],
    cand_texts: np.ndarray,
    source_ids,
    candidate_rows,
    probabilities,
    ce,
    top_n: int = 10,
    ce_batch_size: int = 256,
) -> np.ndarray:
    """Replace each entity's top-N pair scores with cross-encoder probabilities.

    Ranking for the top-N selection uses the incoming (LightGBM) scores with
    the same (score desc, candidate row desc) tie-break as
    tune_decisions._build_groups. Pairs outside the top-N are demoted to
    score * 0.5 so they stay below any realistic decision threshold while
    remaining inside the candidate set.
    """
    probs = np.array(probabilities, dtype=np.float64, copy=True)
    if len(probs) == 0 or top_n <= 0:
        return probs
    groups: dict[str, list[int]] = {}
    for i, sid in enumerate(source_ids):
        groups.setdefault(sid, []).append(i)
    pair_texts: list[tuple[str, str]] = []
    selected: list[int] = []
    for sid, indices in groups.items():
        indices.sort(key=lambda i: (-probs[i], -int(candidate_rows[i])))
        for i in indices[:top_n]:
            selected.append(i)
            pair_texts.append((s1_texts[sid], cand_texts[int(candidate_rows[i])]))
    if not pair_texts:
        return probs
    scores = np.asarray(
        ce.predict(pair_texts, batch_size=ce_batch_size, show_progress_bar=False),
        dtype=np.float64,
    ).reshape(-1)
    for i, score in zip(selected, scores):
        probs[i] = float(score)
    if len(selected) < len(probs):
        mask = np.ones(len(probs), dtype=bool)
        mask[np.asarray(selected, dtype=np.int64)] = False
        probs[mask] *= 0.5
    return probs
