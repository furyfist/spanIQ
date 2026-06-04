import numpy as np

from spaniq.statistical.embeddings import cosine_similarity, embed


def test_same_text_high_similarity():
    text = "the refund policy allows returns within 30 days"
    vecs = embed([text, text])
    assert cosine_similarity(vecs[0], vecs[1]) > 0.99


def test_different_text_lower_similarity():
    vecs = embed(["the sky is blue", "quantum computing uses qubits"])
    assert cosine_similarity(vecs[0], vecs[1]) < 0.8


def test_deterministic():
    texts = ["hello world", "foo bar baz"]
    vecs1 = embed(texts)
    vecs2 = embed(texts)
    np.testing.assert_array_equal(vecs1, vecs2)


def test_batch_shape():
    texts = ["one", "two", "three"]
    vecs = embed(texts)
    assert vecs.shape == (3, 384)


def test_embeddings_normalized():
    vecs = embed(["normalize this sentence please"])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-5
