import time
import numpy as np

class ObjectMemory:
    """
    Stores long-term memory of a visual object with decay support.
    """

    def __init__(self, embedding, first_seen=None, last_seen=None,
                 seen_count=1, stability=0.5):
        self.embedding = embedding
        self.first_seen = first_seen or time.time()
        self.last_seen = last_seen or time.time()
        self.seen_count = seen_count
        self.stability = stability

    def update(self, new_embedding):
        self.embedding = 0.8 * self.embedding + 0.2 * new_embedding
        self.last_seen = time.time()
        self.seen_count += 1
        self.stability = min(1.0, self.stability + 0.1)

    def decay_score(self, now=None):
        if now is None:
            now = time.time()

        time_gap = now - self.last_seen
        time_factor = np.exp(-time_gap / 10.0)
        return time_factor * self.stability

    # ---------- Persistence ----------
    def to_dict(self):
        return {
            "embedding": self.embedding.tolist(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "seen_count": self.seen_count,
            "stability": self.stability
        }

    @staticmethod
    def from_dict(data):
        return ObjectMemory(
            embedding=np.array(data["embedding"]),
            first_seen=data["first_seen"],
            last_seen=data["last_seen"],
            seen_count=data["seen_count"],
            stability=data["stability"]
        )
