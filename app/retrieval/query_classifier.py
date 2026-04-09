from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryRoute:
    route: str
    reason: str


class QueryClassifier:
    """Heuristic query router.

    direct: simple, likely evergreen/general knowledge queries
    rag: needs external evidence
    """

    TIME_SENSITIVE = {"latest", "current", "today", "recent", "recently", "updated", "new", "now"}
    COMPLEX_CUES = {"compare", "explain", "why", "how", "relationship", "tradeoff", "multi-hop", "evidence"}

    def classify(self, question: str) -> QueryRoute:
        q = question.lower().strip()
        if any(tok in q for tok in self.TIME_SENSITIVE):
            return QueryRoute(route="rag", reason="time-sensitive query")
        if any(tok in q for tok in self.COMPLEX_CUES):
            return QueryRoute(route="rag", reason="complex reasoning query")
        if len(re.findall(r"\w+", q)) > 14:
            return QueryRoute(route="rag", reason="long query likely needs retrieval")
        if any(c in q for c in [":", ";"]):
            return QueryRoute(route="rag", reason="structured query")
        return QueryRoute(route="direct", reason="simple query")
