import numpy as np

def cosine_similarity(a, b):
    """
    Computes cosine similarity between two vectors.
    Returns a value between -1 and 1.
    """
    if a is None or b is None:
        return 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
