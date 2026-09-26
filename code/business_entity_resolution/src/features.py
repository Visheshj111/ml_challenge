"""Pair features for the ranked candidate set.

Features are computed only for retrieved pairs and prepared records are cached
within each batch. This avoids re-normalising all 10M candidate rows for every
inference batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blocking import CandidateHit
from preprocess import prepare_record

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # Requirements pin RapidFuzz for full-scale use.
    raise RuntimeError("rapidfuzz is required for feature extraction") from exc

try:
    import jellyfish
except ImportError:
    jellyfish = None


FEATURE_NAMES = [
    "name_raw_exact", "name_norm_exact", "name_core_exact", "name_raw_wratio",
    "name_norm_lev_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaro_winkler", "name_token_jaccard", "name_initials_match",
    "name_legal_type_match", "name_length_ratio", "name_token_count_difference",
    "addr_raw_wratio", "addr_norm_lev_ratio", "addr_token_sort_ratio",
    "addr_token_set_ratio", "addr_token_jaccard", "addr_length_ratio",
    "numbers_exact", "numbers_overlap", "numbers_conflict", "number_count_difference",
    "country_match", "country_conflict", "country_missing_either",
    "name_strong_and_addr_strong", "name_strong_and_number_conflict",
    "retrieval_score", "lexical_score", "lexical_pass_fraction", "lexical_rank_inverse",
    "dense_score", "dense_rank_inverse", "retrieved_by_both",
]


def _tokens(value: str) -> set[str]:
    return set(value.split()) if value else set()


def _jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _length_ratio(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    return min(len(left), len(right)) / longest if longest else 1.0


def _number_features(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[float, float, float, float]:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0, 1.0, 0.0, 0.0
    if not left_set or not right_set:
        return 0.0, 0.0, 0.0, 1.0
    overlap = len(left_set & right_set) / len(left_set | right_set)
    exact = float(left_set == right_set)
    # A fully disjoint number set is strong negative evidence. Partial mismatch
    # remains a softer signal because addresses can contain PINs and suites.
    conflict = float(not (left_set & right_set))
    count_diff = abs(len(left_set) - len(right_set)) / max(len(left_set), len(right_set), 1)
    return exact, overlap, conflict, count_diff


def _jaro_winkler(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if jellyfish is None:
        return fuzz.ratio(left, right) / 100.0
    return float(jellyfish.jaro_winkler_similarity(left, right))


def extract_pair_features(left: dict[str, Any], right: dict[str, Any], hit: CandidateHit) -> np.ndarray:
    name_raw = fuzz.WRatio(str(left["name_raw"]), str(right["name_raw"])) / 100.0
    name_lev = fuzz.ratio(str(left["name_norm"]), str(right["name_norm"])) / 100.0
    name_sort = fuzz.token_sort_ratio(str(left["name_core"]), str(right["name_core"])) / 100.0
    name_set = fuzz.token_set_ratio(str(left["name_core"]), str(right["name_core"])) / 100.0
    name_jaccard = _jaccard(str(left["name_core"]), str(right["name_core"]))
    address_raw = fuzz.WRatio(str(left["addr_raw"]), str(right["addr_raw"])) / 100.0
    address_lev = fuzz.ratio(str(left["addr_norm"]), str(right["addr_norm"])) / 100.0
    address_sort = fuzz.token_sort_ratio(str(left["addr_norm"]), str(right["addr_norm"])) / 100.0
    address_set = fuzz.token_set_ratio(str(left["addr_norm"]), str(right["addr_norm"])) / 100.0
    address_jaccard = _jaccard(str(left["addr_norm"]), str(right["addr_norm"]))
    numbers_exact, numbers_overlap, numbers_conflict, number_count_difference = _number_features(
        left["addr_numbers"], right["addr_numbers"]
    )
    country_left, country_right = str(left["country"]), str(right["country"])
    country_missing = float(not country_left or not country_right)
    country_match = float(bool(country_left) and country_left == country_right)
    country_conflict = float(bool(country_left and country_right and country_left != country_right))
    name_strong = float(name_set >= 0.90 or name_lev >= 0.94)
    address_strong = float(address_set >= 0.85 or numbers_exact == 1.0)

    return np.array([
        float(left["name_raw"] == right["name_raw"] and bool(left["name_raw"])),
        float(left["name_norm"] == right["name_norm"] and bool(left["name_norm"])),
        float(left["name_core"] == right["name_core"] and bool(left["name_core"])),
        name_raw, name_lev, name_sort, name_set,
        _jaro_winkler(str(left["name_core"]), str(right["name_core"])),
        name_jaccard,
        float(bool(left["name_initials"]) and left["name_initials"] == right["name_initials"]),
        float(bool(left["legal_type"]) and left["legal_type"] == right["legal_type"]),
        _length_ratio(str(left["name_core"]), str(right["name_core"])),
        abs(len(str(left["name_core"]).split()) - len(str(right["name_core"]).split())) /
        max(len(str(left["name_core"]).split()), len(str(right["name_core"]).split()), 1),
        address_raw, address_lev, address_sort, address_set, address_jaccard,
        _length_ratio(str(left["addr_norm"]), str(right["addr_norm"])),
        numbers_exact, numbers_overlap, numbers_conflict, number_count_difference,
        country_match, country_conflict, country_missing,
        name_strong * address_strong, name_strong * numbers_conflict,
        hit.retrieval_score,
        min(hit.lexical_score / 100.0, 2.0),
        min(hit.lexical_pass_count / 11.0, 1.0),
        (1.0 / hit.lexical_rank) if hit.lexical_rank else 0.0,
        hit.dense_score,
        (1.0 / hit.dense_rank) if hit.dense_rank else 0.0,
        float(bool(hit.lexical_rank) and bool(hit.dense_rank)),
    ], dtype=np.float32)


def extract_features_for_hits(
    s1_batch,
    candidates,
    hits_by_s1: dict[str, list[CandidateHit]],
) -> tuple[np.ndarray, list[str], list[int], list[CandidateHit]]:
    """Build a feature matrix for one bounded Source 1 batch."""
    candidate_cache: dict[int, dict[str, Any]] = {}
    rows: list[np.ndarray] = []
    source_ids: list[str] = []
    candidate_rows: list[int] = []
    hits_out: list[CandidateHit] = []

    for s1_id, name, address, country in s1_batch[["entity_id", "business_name", "business_address", "country"]].itertuples(index=False, name=None):
        left = prepare_record(name, address, country)
        for hit in hits_by_s1.get(s1_id, []):
            right = candidate_cache.get(hit.row)
            if right is None:
                candidate = candidates.iloc[hit.row]
                right = prepare_record(candidate.business_name, candidate.business_address, candidate.country)
                candidate_cache[hit.row] = right
            rows.append(extract_pair_features(left, right, hit))
            source_ids.append(s1_id)
            candidate_rows.append(hit.row)
            hits_out.append(hit)

    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), [], [], []
    return np.vstack(rows), source_ids, candidate_rows, hits_out
