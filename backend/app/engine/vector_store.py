import math
import re
from typing import List, Dict

class VectorStore:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 2]
        subwords = []
        for t in tokens:
            subwords.append(t)
            if len(t) >= 4:
                for i in range(len(t) - 2):
                    subwords.append(t[i:i+3])
        return subwords

    def embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1

        for t, c in counts.items():
            idx = abs(hash(t)) % self.dimension
            tf = 1.0 + math.log(c)
            vector[idx] += tf

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]
        return vector

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(0.0, min(1.0, float(dot)))

vector_store = VectorStore()
