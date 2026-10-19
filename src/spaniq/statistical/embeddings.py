import os
import threading
from collections import OrderedDict

import numpy as np

# Override with SPANIQ_EMBEDDING_MODEL or set_model(); a local path works offline.
MODEL_NAME = os.environ.get("SPANIQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_model = None
_lock = threading.Lock()
_reference_cache: OrderedDict = OrderedDict()
_REFERENCE_CACHE_SIZE = 64


def set_model(name: str) -> None:
    """Switch the embedding model; the next embed() call loads it."""
    global MODEL_NAME, _model
    with _lock:
        MODEL_NAME = name
        _model = None
        _reference_cache.clear()


def get_model():
    """Load the model once, even when several evaluate() threads ask at the same time."""
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL_NAME)
        return _model


def embed(texts: list[str]) -> np.ndarray:
    """Encode texts to normalized embeddings. Returns (n, dim) float32 array."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_reference(texts: list[str]) -> np.ndarray:
    """Embeddings of a fixed reference set such as a baseline, cached across calls.

    Monitoring scores every trace against the same baseline, so re-encoding it
    each time made per-trace cost grow with baseline size. The set is encoded
    as its own batch, so the result never depends on what else is being scored.
    """
    key = (MODEL_NAME, tuple(texts))
    with _lock:
        hit = _reference_cache.get(key)
        if hit is not None:
            _reference_cache.move_to_end(key)
            return hit
    vecs = embed(list(texts))
    vecs.setflags(write=False)
    with _lock:
        _reference_cache[key] = vecs
        while len(_reference_cache) > _REFERENCE_CACHE_SIZE:
            _reference_cache.popitem(last=False)
    return vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized embedding vectors."""
    return float(np.dot(a, b))
