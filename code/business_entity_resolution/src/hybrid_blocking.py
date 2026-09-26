"""Hybrid candidate retrieval: lexical blocking + dense multilingual embeddings.

Retrieval strategy
------------------
1. Lexical pass (``blocking.CandidateIndex``): 11 country-prefixed hash indexes.
2. Dense pass (``DenseIndex``): multilingual sentence embeddings, exact top-k
   cosine search restricted to the query's country segment. The multilingual
   encoder is what makes cross-script matches (Devanagari candidate vs Latin
   source name) retrievable at all.
3. Merge: reciprocal-rank fusion; the fused top-k becomes the final candidate set.

Memory model
------------
Dense embeddings never live fully in RAM. They are encoded once into an fp16
memmap cache on disk (per data directory, under ``<model_dir>/cache/``); at
search time each country segment is streamed through the compute device in
bounded SEARCH_CHUNK blocks, so peak device memory is one chunk (~200MB)
regardless of segment size (the largest country alone is 4.4GB fp16, which
fits neither a 6GB GPU next to the encoder nor spare RAM next to the source
DataFrames). There is no FAISS dependency: top-k is an exact matmul, which
also keeps results deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import (
    CandidateIndex,
    CandidateHit,
    _fingerprint,
    candidate_ids,
    iter_prepared,
    load_sources,
    record_key_values,
)
from blocking import retrieve_batch as lexical_retrieve_batch
from preprocess import prepare_record

try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAVE_DENSE = True
except ImportError:
    torch = None
    SentenceTransformer = None
    HAVE_DENSE = False


DEFAULT_TOP_K = 60            # final merged candidates per S1 entity
DEFAULT_LEXICAL_K = 100       # lexical pool size before the merge
DEFAULT_DENSE_K = 40          # dense pool size before the merge
DEFAULT_ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ENCODE_BATCH = 256
ENCODER_TEXT_VERSION = 1
ENCODE_CHUNK = 200_000        # texts per encode call (bounds the fp32 output array)
SEARCH_CHUNK = 250_000        # segment rows per matmul (bounds the sims matrix)
MIN_SEARCH_CHUNK = 15_625     # adaptive CUDA OOM floor (12MB fp16 candidate block)
QUERY_GROUP = 256             # queries per matmul (bounds the sims matrix)
RRF_K = 60.0


def encoder_text(prepared: dict) -> str:
    """The text fed to the dense encoder; the name is repeated to weight it."""
    name = str(prepared["name_core"]) or str(prepared["name_raw"])
    return f"{name} {name} {prepared['addr_norm']}".strip()


def source_paths(data_dir: str) -> list[str]:
    prefix = Path(data_dir).name
    return [os.path.join(data_dir, f"{prefix}_source2.tsv"), os.path.join(data_dir, f"{prefix}_source3.tsv")]


def _cache_fingerprint(files: list[str], encoder: str, cand_limit: int | None) -> str:
    parts = [f"encoder={encoder}", f"text_version={ENCODER_TEXT_VERSION}", f"cand_limit={cand_limit}"]
    for path in files:
        if os.path.isfile(path):
            stat = os.stat(path)
            parts.append(f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime)}")
        else:
            parts.append(f"{os.path.basename(path)}:missing")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DenseIndex:
    """Country-segmented exact top-k cosine index over an fp16 memmap cache."""

    def __init__(self, cache_dir: str, encoder: str, device: str, encode_batch: int):
        self.cache_dir = cache_dir
        self.encoder = encoder
        self.encode_batch = encode_batch
        self.device = _resolve_device(device)
        self.device_is_cuda = self.device.startswith("cuda")
        self.dim: int = 0
        self.count: int = 0
        self.segments: dict[str, tuple[int, int]] = {}  # country -> (start, count)
        self.order: np.ndarray | None = None            # memmap position -> candidate row
        self._memmap = None
        self._model = None

    # -- construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        candidates: pd.DataFrame,
        texts_by_country: dict[int, list[str]],
        country_ids: np.ndarray,
        country_names: dict[int, str],
        cache_dir: str,
        encoder: str,
        device: str,
        encode_batch: int,
    ) -> "DenseIndex":
        """Encode texts (already grouped by country) into a fresh cache."""
        os.makedirs(cache_dir, exist_ok=True)
        index = cls(cache_dir, encoder, device, encode_batch)
        index._load_model()

        n = len(candidates)
        order = np.argsort(country_ids, kind="stable")
        segments: dict[str, tuple[int, int]] = {}
        position = 0
        for cid in range(len(country_names)):
            count = len(texts_by_country[cid])
            if count:
                segments[country_names[cid]] = (position, count)
                position += count

        print(f"[dense] Encoding {n:,} candidate texts on device '{index.device}' "
              f"(model={encoder}, dim={index.dim}) ...")
        memmap_path = os.path.join(cache_dir, "embeddings.fp16.npy")
        memmap = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float16, shape=(n, index.dim))
        for cid in range(len(country_names)):
            country = country_names[cid]
            start, count = segments.get(country, (0, 0))
            texts = texts_by_country[cid]
            for chunk_start in range(0, count, ENCODE_CHUNK):
                chunk = texts[chunk_start:chunk_start + ENCODE_CHUNK]
                vectors = index._encode_texts(chunk)
                memmap[start + chunk_start:start + chunk_start + len(chunk)] = vectors.astype(np.float16)
                done = min(chunk_start + len(chunk), count)
                print(f"[dense]   {country or '<empty>'}: encoded {done:,}/{count:,}")
            texts_by_country[cid] = None  # free the text list as we go
        memmap.flush()
        del memmap

        order_path = os.path.join(cache_dir, "order.npy")
        np.save(order_path, order.astype(np.int64))
        meta = {
            "encoder": encoder,
            "text_version": ENCODER_TEXT_VERSION,
            "dim": index.dim,
            "count": n,
            "segments": {country: [int(s), int(c)] for country, (s, c) in segments.items()},
        }
        with open(os.path.join(cache_dir, "dense_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        index.order = order.astype(np.int64)
        index.segments = segments
        index.count = n
        print(f"[dense] Cache written to {cache_dir}")
        return index

    @classmethod
    def load(cls, cache_dir: str, encoder: str, device: str, encode_batch: int) -> "DenseIndex | None":
        meta_path = os.path.join(cache_dir, "dense_meta.json")
        order_path = os.path.join(cache_dir, "order.npy")
        memmap_path = os.path.join(cache_dir, "embeddings.fp16.npy")
        if not (os.path.isfile(meta_path) and os.path.isfile(order_path) and os.path.isfile(memmap_path)):
            return None
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        index = cls(cache_dir, meta["encoder"], device, encode_batch)
        index.dim = int(meta["dim"])
        index.count = int(meta["count"])
        index.segments = {country: (int(start), int(count)) for country, (start, count) in meta["segments"].items()}
        index.order = np.load(order_path)
        index._load_model()
        print(f"[dense] Reusing cached embeddings: {index.count:,} vectors, "
              f"{len(index.segments)} countries, dim={index.dim}")
        return index

    def _load_model(self):
        if not HAVE_DENSE:
            raise RuntimeError("sentence-transformers/torch not installed; dense retrieval unavailable")
        print(f"[dense] Loading encoder '{self.encoder}' on device '{self.device}' ...")
        self._model = SentenceTransformer(self.encoder, device=self.device)
        if self.device_is_cuda:
            try:
                self._model.half()
            except Exception:
                pass
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = int(self._model.get_embedding_dimension())
        else:  # sentence-transformers < 5
            self.dim = int(self._model.get_sentence_embedding_dimension())

    # -- search -----------------------------------------------------------

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.encode_batch,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _open_memmap(self) -> np.ndarray:
        if self._memmap is None:
            self._memmap = np.load(os.path.join(self.cache_dir, "embeddings.fp16.npy"), mmap_mode="r")
        return self._memmap

    def _score_chunk(
        self,
        queries: "torch.Tensor",
        source: np.ndarray,
        segment_start: int,
        segment_count: int,
        chunk_start: int,
        chunk_size: int,
        k: int,
        group_bounds: list[tuple[int, int]],
        dtype: "torch.dtype",
    ) -> tuple[int, list[tuple["torch.Tensor", "torch.Tensor"]]]:
        chunk_end = min(chunk_start + chunk_size, segment_count)
        block = np.array(source[segment_start + chunk_start:segment_start + chunk_end])
        chunk = torch.from_numpy(block).to(device=self.device, dtype=dtype)
        chunk_results = []
        for group_start, group_end in group_bounds:
            sims = queries[group_start:group_end] @ chunk.t()
            chunk_k = min(k, chunk_end - chunk_start)
            scores, positions = torch.topk(sims, chunk_k, dim=1)
            chunk_results.append((scores, positions + chunk_start))
        return chunk_end, chunk_results

    def search(self, s1_batch: pd.DataFrame, top_k: int) -> dict[str, list[tuple[int, float, int]]]:
        """Dense top-k per S1 entity: entity_id -> [(candidate row, cosine, rank)].

        The country segment is streamed from the fp16 memmap in SEARCH_CHUNK
        blocks rather than loaded whole, so peak device memory is one chunk no
        matter how large the segment is. Each chunk is scored against every
        query of the country before the next chunk is fetched; repeated passes
        over a segment are served from the OS page cache.
        """
        if self._model is None:
            return {}
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for entity_id, name, address, country in s1_batch[
            ["entity_id", "business_name", "business_address", "country"]
        ].itertuples(index=False, name=None):
            prepared = prepare_record(name, address, country)
            groups[str(prepared["country"])].append((entity_id, encoder_text(prepared)))

        results: dict[str, list[tuple[int, float, int]]] = {}
        source = self._open_memmap()
        for country, items in groups.items():
            segment = self.segments.get(country)
            if segment is None:
                continue
            start, count = segment
            k = min(top_k, count)
            if k <= 0:
                continue
            dtype = torch.float16 if self.device_is_cuda else torch.float32
            queries = torch.from_numpy(
                self._encode_texts([text for _, text in items])
            ).to(device=self.device, dtype=dtype)
            group_bounds = [
                (offset, min(offset + QUERY_GROUP, len(items)))
                for offset in range(0, len(items), QUERY_GROUP)
            ]
            best: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(group_bounds)
            chunk_size = SEARCH_CHUNK
            chunk_start = 0
            while chunk_start < count:
                try:
                    chunk_end, chunk_results = self._score_chunk(
                        queries, source, start, count, chunk_start, chunk_size,
                        k, group_bounds, dtype,
                    )
                except torch.cuda.OutOfMemoryError:
                    if not self.device_is_cuda or chunk_size <= MIN_SEARCH_CHUNK:
                        raise
                    chunk_size = max(MIN_SEARCH_CHUNK, chunk_size // 2)
                    torch.cuda.empty_cache()
                    print(f"[dense] CUDA OOM; retrying with search chunk {chunk_size:,}")
                    continue
                for group_index, (scores_chunk, positions_chunk) in enumerate(chunk_results):
                    current = best[group_index]
                    if current is None:
                        best[group_index] = (scores_chunk, positions_chunk)
                        continue
                    scores, positions = current
                    scores = torch.cat([scores, scores_chunk], dim=1)
                    positions = torch.cat([positions, positions_chunk], dim=1)
                    if scores.shape[1] > k:
                        scores, merge_order = torch.topk(scores, k, dim=1)
                        positions = torch.gather(positions, 1, merge_order)
                    best[group_index] = (scores, positions)
                chunk_start = chunk_end
            del queries
            if self.device_is_cuda:
                torch.cuda.empty_cache()
            print(f"[dense] Searched segment {country!r}: {count:,} vectors x {len(items):,} queries")
            for group_index, (group_start, group_end) in enumerate(group_bounds):
                scores, positions = best[group_index]
                scores_np = scores.float().cpu().numpy()
                positions_np = positions.cpu().numpy()
                for row_index in range(group_end - group_start):
                    entity_id = items[group_start + row_index][0]
                    hits = []
                    for rank in range(scores_np.shape[1]):
                        position = int(positions_np[row_index, rank])
                        candidate_row = int(self.order[start + position])
                        hits.append((candidate_row, float(scores_np[row_index, rank]), rank + 1))
                    results[entity_id] = hits
        return results


def merge_hits(
    lexical_hits: list[CandidateHit],
    dense_hits: list[tuple[int, float, int]],
    top_k: int,
) -> list[CandidateHit]:
    """Reciprocal-rank fusion of the two retrieval modes."""
    lexical_by_row = {hit.row: hit for hit in lexical_hits}
    dense_by_row = {row: (score, rank) for row, score, rank in dense_hits}
    fused: list[tuple[float, int, CandidateHit | None, float, int]] = []
    for row in set(lexical_by_row) | set(dense_by_row):
        lexical = lexical_by_row.get(row)
        dense_score, dense_rank = dense_by_row.get(row, (0.0, 0))
        score = 0.0
        if lexical is not None and lexical.lexical_rank:
            score += 1.0 / (RRF_K + lexical.lexical_rank)
        if dense_rank:
            score += 1.0 / (RRF_K + dense_rank)
        fused.append((score, row, lexical, dense_score, dense_rank))
    fused.sort(key=lambda item: (-item[0], item[1]))
    return [
        CandidateHit(
            row=row,
            retrieval_score=float(score),
            lexical_score=(lexical.lexical_score if lexical is not None else 0.0),
            lexical_pass_count=(lexical.lexical_pass_count if lexical is not None else 0),
            lexical_rank=(lexical.lexical_rank if lexical is not None else 0),
            dense_score=float(dense_score),
            dense_rank=int(dense_rank),
        )
        for score, row, lexical, dense_score, dense_rank in fused[:top_k]
    ]


class HybridIndex:
    """Lexical index + optional dense index, merged per S1 entity."""

    def __init__(self, lexical: CandidateIndex, dense: DenseIndex | None,
                 top_k: int, lexical_k: int, dense_k: int):
        self.lexical = lexical
        self.dense = dense
        self.top_k = top_k
        self.lexical_k = lexical_k
        self.dense_k = dense_k

    @classmethod
    def build(
        cls,
        candidates: pd.DataFrame,
        data_dir: str,
        model_dir: str = "models",
        top_k: int = DEFAULT_TOP_K,
        lexical_k: int = DEFAULT_LEXICAL_K,
        dense_k: int = DEFAULT_DENSE_K,
        encoder: str = DEFAULT_ENCODER,
        device: str = "auto",
        encode_batch: int = DEFAULT_ENCODE_BATCH,
        dense_enabled: bool = True,
        cand_limit: int | None = None,
    ) -> "HybridIndex":
        if cand_limit is not None:
            candidates = candidates.head(cand_limit).copy()
            print(f"[hybrid] cand_limit={cand_limit}: using {len(candidates):,} candidate rows")
        if dense_enabled and not HAVE_DENSE:
            print("[hybrid] WARNING: torch/sentence-transformers unavailable - "
                  "falling back to lexical-only retrieval.")
            dense_enabled = False

        cache_dir = None
        dense_index = None
        fingerprint = None
        if dense_enabled:
            prefix = Path(data_dir).name
            suffix = f"_sub{cand_limit}" if cand_limit is not None else ""
            cache_dir = os.path.join(model_dir, "cache", f"{prefix}_dense{suffix}")
            fingerprint = _cache_fingerprint(source_paths(data_dir), encoder, cand_limit)
            meta_path = os.path.join(cache_dir, "dense_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as handle:
                    stored = json.load(handle)
                if (stored.get("fingerprint") == fingerprint
                        and stored.get("encoder") == encoder
                        and stored.get("text_version") == ENCODER_TEXT_VERSION):
                    dense_index = DenseIndex.load(cache_dir, encoder, device, encode_batch)
                else:
                    print(f"[dense] Cache fingerprint mismatch - re-encoding into {cache_dir}")

        n_rows = len(candidates)
        print(f"[hybrid] Building indexes for {n_rows:,} S2/S3 records "
              f"(lexical_k={lexical_k}, dense_k={dense_k}, top_k={top_k}) ...")
        key_arrays = [np.empty(n_rows, dtype=np.uint64) for _ in range(11)]
        need_dense = dense_enabled and dense_index is None
        country_ids = np.empty(n_rows, dtype=np.int32) if need_dense else None
        country_names: dict[int, str] = {}
        country_to_cid: dict[str, int] = {}
        texts_by_country: dict[int, list[str]] = defaultdict(list) if need_dense else {}

        for row_idx, prepared in iter_prepared(candidates):
            for array, value in zip(key_arrays, record_key_values(prepared)):
                array[row_idx] = _fingerprint(value)
            if need_dense:
                country = str(prepared["country"])
                cid = country_to_cid.get(country)
                if cid is None:
                    cid = len(country_names)
                    country_to_cid[country] = cid
                    country_names[cid] = country
                country_ids[row_idx] = cid
                texts_by_country[cid].append(encoder_text(prepared))
            if (row_idx + 1) % 500_000 == 0:
                print(f"[hybrid]   prepared {row_idx + 1:,}/{n_rows:,} candidate rows")

        lexical = CandidateIndex.from_key_arrays(key_arrays)

        if need_dense:
            dense_index = DenseIndex.build(
                candidates, texts_by_country, country_ids, country_names,
                cache_dir, encoder, device, encode_batch,
            )
            meta_path = os.path.join(cache_dir, "dense_meta.json")
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            meta["fingerprint"] = fingerprint
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
        return cls(lexical, dense_index, top_k, lexical_k, dense_k)

    def retrieve_batch(self, s1_batch: pd.DataFrame, top_k: int | None = None) -> dict[str, list[CandidateHit]]:
        top_k = self.top_k if top_k is None else top_k
        lexical = lexical_retrieve_batch(s1_batch, self.lexical, top_k=self.lexical_k)
        if self.dense is None:
            return {s1_id: hits[:top_k] for s1_id, hits in lexical.items()}
        dense = self.dense.search(s1_batch, self.dense_k)
        return {
            s1_id: merge_hits(lexical.get(s1_id, []), dense.get(s1_id, []), top_k)
            for s1_id in s1_batch["entity_id"]
        }


def retrieve_batch(s1_batch: pd.DataFrame, index: HybridIndex, top_k: int | None = None) -> dict[str, list[CandidateHit]]:
    """Drop-in replacement for the lexical retrieve_batch used by train/predict."""
    return index.retrieve_batch(s1_batch, top_k=top_k)


def iter_country_batches(s1: pd.DataFrame, batch_size: int = 1_000):
    """Yield S1 batches whose rows share one country (raw column value order).

    Country-grouped iteration keeps each dense search() call inside a single
    country segment, so the memmap reads stay sequential and page-cache
    friendly.
    """
    if len(s1) == 0:
        return
    country_values = s1["country"].astype(str).to_numpy()
    order = np.argsort(country_values, kind="stable")
    sorted_s1 = s1.iloc[order]
    for start in range(0, len(sorted_s1), batch_size):
        yield sorted_s1.iloc[start:start + batch_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic: hybrid retrieval for one data directory")
    parser.add_argument("--data-dir", default="dataset/test")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--lexical-k", type=int, default=DEFAULT_LEXICAL_K)
    parser.add_argument("--dense-k", type=int, default=DEFAULT_DENSE_K)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch", type=int, default=DEFAULT_ENCODE_BATCH)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--s1-limit", type=int, default=100)
    args = parser.parse_args()

    s1, s2, s3 = load_sources(args.data_dir, s1_limit=args.s1_limit)
    candidates = pd.concat([s2, s3], ignore_index=True)
    index = HybridIndex.build(
        candidates, args.data_dir, model_dir=args.model_dir, top_k=args.top_k,
        lexical_k=args.lexical_k, dense_k=args.dense_k, encoder=args.encoder,
        device=args.device, encode_batch=args.encode_batch, dense_enabled=not args.no_dense,
    )
    for batch in iter_country_batches(s1, batch_size=50):
        hits_by_s1 = retrieve_batch(batch, index)
        for s1_id in batch["entity_id"].head(5):
            hits = hits_by_s1[s1_id]
            dense_only = [h for h in hits if h.dense_rank and not h.lexical_rank]
            print(f"[hybrid] {s1_id}: {len(hits)} candidates "
                  f"({len(dense_only)} dense-only) best={hits[0].row if hits else None}")
        break
