import os
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np


@dataclass
class StoredChunk:
    document: str
    page: int
    chunk: str
    chunk_index: int = 0


@dataclass
class SearchResult:
    document: str
    page: int
    chunk: str
    chunk_index: int
    score: float


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: List[StoredChunk]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def from_disk(cls, base_path: str) -> "VectorStore":
        # Expect `index.faiss` and `chunks.npy` or similar artifacts.
        index_path = os.path.join(base_path, "index.faiss")
        meta_path = os.path.join(base_path, "chunks_metadata.npy")

        index = faiss.read_index(index_path)
        # chunks_metadata.npy could be a structured numpy array or pickled list.
        chunks_arr = np.load(meta_path, allow_pickle=True)
        chunks = [StoredChunk(**item) for item in chunks_arr.tolist()]
        return cls(index=index, chunks=chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        query_embedding = query_embedding.astype("float32")[None, :]
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)
        idxs = indices[0]
        dists = distances[0]

        results: List[SearchResult] = []
        for i, dist in zip(idxs, dists):
            if i < 0 or i >= len(self.chunks):
                continue
            chunk = self.chunks[i]
            # FAISS returns smaller distances for closer neighbors; invert to similarity-like score.
            score = float(dist)
            results.append(
                SearchResult(
                    document=chunk.document,
                    page=chunk.page,
                    chunk=chunk.chunk,
                    chunk_index=chunk.chunk_index,
                    score=round(score, 3),
                )
            )
        return results
