import time
import json
import os

from cognition.similarity import cosine_similarity
from cognition.memory import ObjectMemory

SIMILARITY_THRESHOLD = 0.65
DECAY_THRESHOLD = 0.15
CLEANUP_INTERVAL_SEC = 2.0

MEMORY_FILE = "memory_bank.json"


class PersistenceEngine:
    """
    Object identity + long-term memory persistence.
    """

    def __init__(self):
        self.memory_bank = []
        self._last_cleanup = time.time()
        self.load()

    # ---------------- Memory logic ----------------
    def match_or_create(self, embedding):
        now = time.time()

        if now - self._last_cleanup >= CLEANUP_INTERVAL_SEC:
            self._cleanup()
            self._last_cleanup = now

        best_match = None
        best_score = 0.0

        for obj in self.memory_bank:
            score = cosine_similarity(obj.embedding, embedding)
            if score > best_score:
                best_score = score
                best_match = obj

        if best_match and best_score >= SIMILARITY_THRESHOLD:
            best_match.update(embedding)
            return best_match, "known"

        new_obj = ObjectMemory(embedding)
        self.memory_bank.append(new_obj)
        return new_obj, "new"

    def _cleanup(self):
        now = time.time()
        self.memory_bank = [
            obj for obj in self.memory_bank
            if obj.decay_score(now) >= DECAY_THRESHOLD
        ]

    # ---------------- Persistence ----------------
    def save(self):
        with open(MEMORY_FILE, "w") as f:
            json.dump(
                [obj.to_dict() for obj in self.memory_bank],
                f
            )

    def load(self):
        if not os.path.exists(MEMORY_FILE):
            return

        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
            self.memory_bank = [
                ObjectMemory.from_dict(d) for d in data
            ]
