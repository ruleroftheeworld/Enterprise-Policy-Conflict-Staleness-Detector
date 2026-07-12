import numpy as np

from engine.embeddings import embedding_service
from engine.embeddings.embedding_service import (
    FALLBACK_DIMENSION,
    clear_embedding_cache,
    embed_obligations,
    embed_texts,
)
from engine.extraction import extract_obligation
from engine.normalization import normalize_obligation


def _obligation(sentence: str):
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence=sentence,
    )

    assert obligation is not None

    return normalize_obligation(obligation)


def test_empty_text_list():
    vectors, warnings = embed_texts([])

    assert vectors == []
    assert warnings == []


def test_fallback_embedding_is_deterministic(monkeypatch):
    clear_embedding_cache()

    monkeypatch.setattr(
        embedding_service,
        "_load_sentence_transformer",
        lambda model_name: None,
    )

    first, _ = embed_texts(["employees use authentication"])

    clear_embedding_cache()

    second, _ = embed_texts(["employees use authentication"])

    assert first == second


def test_fallback_embedding_dimension(monkeypatch):
    clear_embedding_cache()

    monkeypatch.setattr(
        embedding_service,
        "_load_sentence_transformer",
        lambda model_name: None,
    )

    vectors, warnings = embed_texts(["security teams review firewall rules"])

    assert len(vectors) == 1
    assert len(vectors[0]) == FALLBACK_DIMENSION
    assert warnings


def test_fallback_embedding_is_normalized(monkeypatch):
    clear_embedding_cache()

    monkeypatch.setattr(
        embedding_service,
        "_load_sentence_transformer",
        lambda model_name: None,
    )

    vectors, _ = embed_texts(["security teams review firewall rules"])

    norm = np.linalg.norm(vectors[0])

    assert np.isclose(norm, 1.0)


def test_duplicate_texts_receive_same_embedding(monkeypatch):
    clear_embedding_cache()

    monkeypatch.setattr(
        embedding_service,
        "_load_sentence_transformer",
        lambda model_name: None,
    )

    vectors, _ = embed_texts(
        [
            "employees use authentication",
            "employees use authentication",
        ]
    )

    assert vectors[0] == vectors[1]


def test_embedding_cache_avoids_reencoding(monkeypatch):
    clear_embedding_cache()

    calls = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append(list(texts))

            return np.ones(
                (len(texts), FALLBACK_DIMENSION),
                dtype=np.float32,
            )

    monkeypatch.setattr(
        embedding_service,
        "_load_sentence_transformer",
        lambda model_name: FakeModel(),
    )

    first, _ = embed_texts(["same text"])
    second, _ = embed_texts(["same text"])

    assert first == second
    assert len(calls) == 1


def test_embed_obligations_preserves_original():
    original = _obligation(
        "Employees must use multi-factor authentication for remote access."
    )

    original_embedding = original.embedding.copy()

    embedded, _ = embed_obligations([original])

    assert original.embedding == original_embedding
    assert embedded[0].obligation_id == original.obligation_id
    assert embedded[0].sentence_text == original.sentence_text
    assert len(embedded[0].embedding) > 0


def test_semantically_identical_normalized_obligations_have_same_embedding():
    first = _obligation(
        "Employees must use multi-factor authentication for remote access."
    )

    second = _obligation(
        "Employees shall use multi-factor authentication for remote access."
    )

    embedded, _ = embed_obligations([first, second])

    assert embedded[0].embedding == embedded[1].embedding


def test_required_and_prohibited_same_target_have_same_embedding_text():
    required = _obligation(
        "Employees must use multi-factor authentication for remote access."
    )

    prohibited = _obligation(
        "Employees must not use multi-factor authentication for remote access."
    )

    assert (
        embedding_service._embedding_text(required)
        == embedding_service._embedding_text(prohibited)
    )


def test_batch_embedding_preserves_order(monkeypatch):
    clear_embedding_cache()

    class FakeModel:
        def encode(self, texts, **kwargs):
            vectors = []

            for index, _ in enumerate(texts):
                vector = np.zeros(FALLBACK_DIMENSION, dtype=np.float32)
                vector[index] = 1.0
                vectors.append(vector)

            return np.asarray(vectors)

    monkeypatch.setattr(
        embedding_service,
        "_load_sentence_transformer",
        lambda model_name: FakeModel(),
    )

    vectors, _ = embed_texts(["first", "second", "third"])

    assert vectors[0][0] == 1.0
    assert vectors[1][1] == 1.0
    assert vectors[2][2] == 1.0