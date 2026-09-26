"""Scalable, ranked candidate retrieval for the entity-resolution pipeline.

The old implementation created dense TF-IDF similarity matrices between every
Source 1 record and very large S2/S3 chunks.  This module instead uses multiple
bounded lexical indexes.  It never materialises an S1 x candidate matrix.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import prepare_record


DEFAULT_TOP_K = 100
MAX_BLOCK_SIZE = 150
INDEX_PROGRESS_EVERY = 500_000


def load_sources(data_dir: str, s1_limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = Path(data_dir).name
    read_kwargs = {"sep": "\t", "dtype": str, "keep_default_na": False}
    s1 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source1.tsv"), **read_kwargs)
    s2 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source2.tsv"), **read_kwargs)
    s3 = pd.read_csv(os.path.join(data_dir, f"{prefix}_source3.tsv"), **read_kwargs)
    if s1_limit is not None:
        s1 = s1.head(s1_limit).copy()
        print(f"[blocking] s1_limit={s1_limit}: using {len(s1):,} S1 rows")
    print(f"[blocking] Loaded: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,} rows from '{data_dir}'")
    return s1, s2, s3


def _fingerprint(value: str) -> np.uint64:
    """A process-local 64-bit key. Collisions are extraordinarily unlikely."""
    return np.uint64(hash(value) & 0xFFFFFFFFFFFFFFFF)


def _key(country: str, value: str) -> str:
    # Ground truth is 100% same-country, so every lexical key is prefixed with
    # the country: wrong-country candidates can never enter a posting list.
    if not value:
        return ""
    return f"{country}\x1f{value}" if country else f"\x1f{value}"


def _numeric_key(country: str, number: str, address_anchor: str) -> str:
    if not number or not address_anchor:
        return ""
    return f"{country}\x1f{number}\x1f{address_anchor}"


def _anchor(tokens: tuple[str, ...]) -> str:
    return tokens[0] if tokens else ""


def _second_anchor(tokens: tuple[str, ...]) -> str:
    return tokens[1] if len(tokens) > 1 else ""


def _prefix_anchor(tokens: tuple[str, ...]) -> str:
    token = _anchor(tokens)
    return token[:4] if len(token) >= 4 else token


def _key_inputs(prepared: dict) -> tuple[str, str, str, str, str, str]:
    return (
        str(prepared["country"]),
        _anchor(prepared["name_tokens"]),
        _second_anchor(prepared["name_tokens"]),
        _prefix_anchor(prepared["name_tokens"]),
        _anchor(prepared["addr_tokens"]),
        prepared["addr_numbers"][0] if prepared["addr_numbers"] else "",
    )


def record_key_values(prepared: dict) -> tuple[str, ...]:
    """The 11 country-prefixed lexical keys for one prepared record."""
    country_value, name_anchor, name_second_anchor, name_prefix, addr_anchor, number = _key_inputs(prepared)
    return (
        _key(country_value, str(prepared["name_norm"])),
        _key(country_value, str(prepared["name_core"])),
        _key(country_value, str(prepared["name_compact"])),
        _key(country_value, str(prepared["name_sorted"])),
        _key(country_value, str(prepared["addr_compact"])),
        _key(country_value, name_anchor),
        _key(country_value, name_second_anchor),
        _key(country_value, name_prefix),
        _key(country_value, addr_anchor),
        _numeric_key(country_value, number, addr_anchor),
        _key(country_value, str(prepared["name_initials"])),
    )


def record_pass_specs(prepared: dict) -> tuple[tuple[str, str, float], ...]:
    """(index attribute, key, weight) triples for one prepared query record."""
    country_value, name_anchor, name_second_anchor, name_prefix, addr_anchor, number = _key_inputs(prepared)
    return (
        ("exact_name", _key(country_value, str(prepared["name_norm"])), 100.0),
        ("core_name", _key(country_value, str(prepared["name_core"])), 92.0),
        ("compact_name", _key(country_value, str(prepared["name_compact"])), 88.0),
        ("sorted_name", _key(country_value, str(prepared["name_sorted"])), 84.0),
        ("compact_address", _key(country_value, str(prepared["addr_compact"])), 52.0),
        ("name_anchor", _key(country_value, name_anchor), 45.0),
        ("name_second_anchor", _key(country_value, name_second_anchor), 30.0),
        ("name_prefix", _key(country_value, name_prefix), 24.0),
        ("address_anchor", _key(country_value, addr_anchor), 30.0),
        ("numeric_address", _numeric_key(country_value, number, addr_anchor), 60.0),
        ("initials", _key(country_value, str(prepared["name_initials"])), 12.0),
    )


def iter_prepared(candidates: pd.DataFrame):
    """Yield (row_idx, prepared record) once per candidate row, in row order."""
    rows = candidates[["business_name", "business_address", "country"]].itertuples(index=False, name=None)
    for row_idx, (name, address, country) in enumerate(rows):
        yield row_idx, prepare_record(name, address, country)


@dataclass(frozen=True)
class CandidateHit:
    row: int
    # Combined evidence when produced by hybrid merge; lexical-only evidence
    # when produced directly by CandidateIndex.retrieve.
    retrieval_score: float
    lexical_score: float = 0.0
    lexical_pass_count: int = 0
    lexical_rank: int = 0  # rank in the lexical output; 0 = not lexically retrieved
    dense_score: float = 0.0
    dense_rank: int = 0  # rank in the dense output; 0 = not densely retrieved


@dataclass
class SortedHashIndex:
    hashes: np.ndarray
    rows: np.ndarray

    @classmethod
    def from_hashes(cls, raw_hashes: np.ndarray) -> "SortedHashIndex":
        order = np.argsort(raw_hashes, kind="stable")
        rows = order.astype(np.int32, copy=False)
        hashes = raw_hashes[order]
        return cls(hashes=hashes, rows=rows)

    def lookup_hash(self, fingerprint: np.uint64, limit: int = MAX_BLOCK_SIZE) -> np.ndarray:
        left = int(np.searchsorted(self.hashes, fingerprint, side="left"))
        right = int(np.searchsorted(self.hashes, fingerprint, side="right"))
        size = right - left
        if size <= 0:
            return np.empty(0, dtype=np.int32)
        values = self.rows[left:right]
        if size <= limit:
            return values
        # A common key must not create an unbounded candidate pool. Spread the
        # deterministic sample across the full posting list instead of keeping
        # only the first rows.
        positions = np.linspace(0, size - 1, num=limit, dtype=np.int64)
        return values[positions]


@dataclass
class CandidateIndex:
    exact_name: SortedHashIndex
    core_name: SortedHashIndex
    compact_name: SortedHashIndex
    sorted_name: SortedHashIndex
    compact_address: SortedHashIndex
    name_anchor: SortedHashIndex
    name_second_anchor: SortedHashIndex
    name_prefix: SortedHashIndex
    address_anchor: SortedHashIndex
    numeric_address: SortedHashIndex
    initials: SortedHashIndex

    @classmethod
    def build(cls, candidates: pd.DataFrame) -> "CandidateIndex":
        n_rows = len(candidates)
        print(f"[blocking] Building bounded retrieval indexes for {n_rows:,} S2/S3 records ...")
        key_arrays = [np.empty(n_rows, dtype=np.uint64) for _ in range(11)]
        for row_idx, prepared in iter_prepared(candidates):
            for array, value in zip(key_arrays, record_key_values(prepared)):
                array[row_idx] = _fingerprint(value)
            if (row_idx + 1) % INDEX_PROGRESS_EVERY == 0:
                print(f"[blocking]   indexed {row_idx + 1:,}/{n_rows:,} candidates")
        return cls.from_key_arrays(key_arrays)

    @classmethod
    def from_key_arrays(cls, key_arrays: list[np.ndarray]) -> "CandidateIndex":
        # Consumes key_arrays as it goes so the caller does not need to keep the
        # unsorted and sorted index sets in memory at the same time.
        indexes_reversed = []
        while key_arrays:
            indexes_reversed.append(SortedHashIndex.from_hashes(key_arrays.pop()))
        indexes = list(reversed(indexes_reversed))
        print("[blocking] Candidate indexes ready.")
        return cls(*indexes)

    def retrieve(self, name: str, address: str, country: str, top_k: int = DEFAULT_TOP_K) -> list[CandidateHit]:
        prepared = prepare_record(name, address, country)
        scores: dict[int, list[float]] = {}
        for field, value, weight in record_pass_specs(prepared):
            if not value:
                continue
            rows = getattr(self, field).lookup_hash(_fingerprint(value))
            for rank, row in enumerate(rows, start=1):
                item = scores.setdefault(int(row), [0.0, 0.0])
                item[0] += weight / np.log2(rank + 1.0)
                item[1] += 1.0

        ranked = sorted(scores.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))[:top_k]
        return [
            CandidateHit(
                row=row,
                retrieval_score=float(values[0]),
                lexical_score=float(values[0]),
                lexical_pass_count=int(values[1]),
                lexical_rank=rank,
            )
            for rank, (row, values) in enumerate(ranked, start=1)
        ]


def retrieve_batch(s1_batch: pd.DataFrame, index: CandidateIndex, top_k: int = DEFAULT_TOP_K) -> dict[str, list[CandidateHit]]:
    result: dict[str, list[CandidateHit]] = {}
    for entity_id, name, address, country in s1_batch[["entity_id", "business_name", "business_address", "country"]].itertuples(index=False, name=None):
        result[entity_id] = index.retrieve(name, address, country, top_k=top_k)
    return result


def candidate_ids(hits: list[CandidateHit], candidate_entity_ids: np.ndarray) -> list[str]:
    return [str(candidate_entity_ids[hit.row]) for hit in hits]


def run_blocking(data_dir: str, output_path: str, top_k: int = DEFAULT_TOP_K, s1_limit: int | None = None) -> None:
    s1, s2, s3 = load_sources(data_dir, s1_limit=s1_limit)
    candidates = pd.concat([s2, s3], ignore_index=True)
    entity_ids = candidates["entity_id"].to_numpy(copy=False)
    index = CandidateIndex.build(candidates)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for start in range(0, len(s1), 2_000):
            batch = s1.iloc[start:start + 2_000]
            hits_by_s1 = retrieve_batch(batch, index, top_k=top_k)
            for s1_id in batch["entity_id"]:
                ids = candidate_ids(hits_by_s1[s1_id], entity_ids)
                handle.write(f"{s1_id}\t{','.join(ids)}\n")
            print(f"[blocking] Retrieved candidates for {min(start + len(batch), len(s1)):,}/{len(s1):,} S1 rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ranked entity-resolution candidates")
    parser.add_argument("--data-dir", default="dataset/test")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--s1-limit", type=int, default=None)
    args = parser.parse_args()
    run_blocking(args.data_dir, args.output, args.top_k, args.s1_limit)
