import numpy as np


def softmax_rows(scores):
    arr = np.asarray(scores, dtype=float)
    max_scores = np.max(arr, axis=-1, keepdims=True)
    shifted = arr - max_scores
    exp_scores = np.exp(shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return weights


def attention_weights(queries, keys):
    q = np.asarray(queries, dtype=float)
    k = np.asarray(keys, dtype=float)
    d_k = q.shape[-1]
    scores = np.dot(q, k.T) / np.sqrt(d_k)
    return softmax_rows(scores)


def scaled_dot_product_attention(queries, keys, values):
    v = np.asarray(values, dtype=float)
    weights = attention_weights(queries, keys)
    return np.dot(weights, v)


if __name__ == "__main__":
    scores = np.arange(12, dtype=float).reshape(3, 4)
    weights = softmax_rows(scores)
    print(f"every row of a softmax adds to one: {weights.sum(axis=-1)}")

    queries = np.arange(12, dtype=float).reshape(3, 4)
    keys = np.arange(20, dtype=float).reshape(5, 4)
    values = np.arange(35, dtype=float).reshape(5, 7)
    out = scaled_dot_product_attention(queries, keys, values)
    print(f"three positions attending over three, four numbers each -> {out.shape}")
