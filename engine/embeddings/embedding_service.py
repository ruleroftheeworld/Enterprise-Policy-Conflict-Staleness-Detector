from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import Counter

import numpy as np

from shared.contracts.policy_analysis import NormalizedObligation


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32

FALLBACK_DIMENSION = 384

_MODEL = None
_MODEL_DEVICE: str | None = None
_MODEL_LOAD_FAILED = False
_MODEL_LOCK = threading.Lock()

_EMBEDDING_CACHE: dict[str, list[float]] = {}
_CACHE_LOCK = threading.Lock()


def _embedding_text(obligation: NormalizedObligation) -> str:
    """
    Build a semantic representation for embedding.

    Do not include modality or negation here.

    REQUIRED and PROHIBITED obligations describing the same semantic
    target must remain close enough to become conflict candidates.
    """
    parts = [
        obligation.subject or "",
        obligation.action or "",
        obligation.object or "",
        obligation.scope or "",
        obligation.frequency or "",
        " ".join(obligation.technology),
        obligation.condition or "",
        obligation.exception or "",
    ]

    return " | ".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )


def _cache_key(model_name: str, text: str) -> str:
    value = f"{model_name}:{text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_sentence_transformer(
    model_name: str = DEFAULT_MODEL_NAME,
):
    global _MODEL
    global _MODEL_DEVICE
    global _MODEL_LOAD_FAILED

    if _MODEL is not None:
        return _MODEL

    if _MODEL_LOAD_FAILED:
        return None

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        if _MODEL_LOAD_FAILED:
            return None

        try:
            from sentence_transformers import SentenceTransformer

            device = _select_device()

            LOGGER.info(
                "Loading embedding model '%s' on %s",
                model_name,
                device,
            )

            try:
                model = SentenceTransformer(
                    model_name,
                    device=device,
                )
                _MODEL_DEVICE = device

            except Exception:
                if device != "cuda":
                    raise

                LOGGER.warning(
                    "CUDA model loading failed. Retrying on CPU.",
                    exc_info=True,
                )

                model = SentenceTransformer(
                    model_name,
                    device="cpu",
                )
                _MODEL_DEVICE = "cpu"

            _MODEL = model

            return _MODEL

        except Exception:
            LOGGER.warning(
                "Embedding model unavailable. "
                "Using deterministic hashing fallback.",
                exc_info=True,
            )

            _MODEL_LOAD_FAILED = True
            return None


def _fallback_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())


def _fallback_embedding(
    text: str,
    dimension: int = FALLBACK_DIMENSION,
) -> list[float]:
    """
    Deterministic signed feature hashing fallback.

    This is not a semantic embedding model. It exists only so the
    deterministic pipeline remains runnable when the transformer model
    is unavailable.
    """
    vector = np.zeros(dimension, dtype=np.float32)

    tokens = _fallback_tokens(text)

    if not tokens:
        return vector.tolist()

    features = list(tokens)

    features.extend(
        f"{tokens[index]}::{tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    )

    counts = Counter(features)

    for feature, count in counts.items():
        digest = hashlib.sha256(feature.encode("utf-8")).digest()

        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] % 2 == 0 else -1.0

        vector[index] += sign * float(count)

    norm = float(np.linalg.norm(vector))

    if norm > 0.0:
        vector /= norm

    return vector.tolist()


def _fallback_embeddings(texts: list[str]) -> list[list[float]]:
    return [_fallback_embedding(text) for text in texts]


def _encode_with_model(
    model,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32).tolist()


def embed_texts(
    texts: list[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[list[float]], list[str]]:
    if not texts:
        return [], []

    warnings: list[str] = []

    results: list[list[float] | None] = [None] * len(texts)

    missing_texts: list[str] = []
    missing_keys: list[str] = []
    missing_positions: dict[str, list[int]] = {}

    for index, text in enumerate(texts):
        key = _cache_key(model_name, text)

        with _CACHE_LOCK:
            cached = _EMBEDDING_CACHE.get(key)

        if cached is not None:
            results[index] = cached.copy()
            continue

        if key not in missing_positions:
            missing_texts.append(text)
            missing_keys.append(key)
            missing_positions[key] = []

        missing_positions[key].append(index)

    if missing_texts:
        model = _load_sentence_transformer(model_name)

        if model is None:
            generated = _fallback_embeddings(missing_texts)

            warnings.append(
                "Sentence-transformer embedding model unavailable; "
                "used deterministic hashing fallback."
            )

        else:
            try:
                generated = _encode_with_model(
                    model,
                    missing_texts,
                    batch_size,
                )

            except Exception:
                LOGGER.warning(
                    "Embedding inference failed. "
                    "Using deterministic hashing fallback.",
                    exc_info=True,
                )

                generated = _fallback_embeddings(missing_texts)

                warnings.append(
                    "Sentence-transformer inference failed; "
                    "used deterministic hashing fallback."
                )

        for key, embedding in zip(missing_keys, generated):
            with _CACHE_LOCK:
                _EMBEDDING_CACHE[key] = embedding.copy()

            for position in missing_positions[key]:
                results[position] = embedding.copy()

    final_results = []

    for result in results:
        if result is None:
            raise RuntimeError("Embedding generation produced an incomplete result.")

        final_results.append(result)

    return final_results, warnings


def embed_obligations(
    obligations: list[NormalizedObligation],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[NormalizedObligation], list[str]]:
    if not obligations:
        return [], []

    embedded = [
        obligation.model_copy(deep=True)
        for obligation in obligations
    ]

    texts = [
        _embedding_text(obligation)
        for obligation in embedded
    ]

    vectors, warnings = embed_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
    )

    for obligation, vector in zip(embedded, vectors):
        obligation.embedding = vector

    return embedded, warnings


def get_embedding_backend() -> str:
    if _MODEL is not None:
        return f"sentence-transformers:{_MODEL_DEVICE}"

    if _MODEL_LOAD_FAILED:
        return "deterministic-hashing-fallback"

    return "not-loaded"


def clear_embedding_cache() -> None:
    with _CACHE_LOCK:
        _EMBEDDING_CACHE.clear()


def reset_embedding_service_for_tests() -> None:
    global _MODEL
    global _MODEL_DEVICE
    global _MODEL_LOAD_FAILED

    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_DEVICE = None
        _MODEL_LOAD_FAILED = False

    clear_embedding_cache()