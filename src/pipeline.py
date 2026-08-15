"""
Full lightweight detection pipeline: Sanitization -> Keyword/Pattern
Analysis -> Contextual Risk Scoring -> Decision.
"""
import time
from dataclasses import dataclass

from src.sanitizer import sanitize
from src.keyword_engine import analyze
from src.risk_scorer import RiskScorer, extract_features


@dataclass
class PipelineResult:
    decision: str
    score: float
    latency_ms: float


class InjectionDetectionPipeline:
    def __init__(self, scorer: RiskScorer):
        self.scorer = scorer

    def run(self, raw_text: str, source_trust: float = 0.0) -> PipelineResult:
        t0 = time.perf_counter()
        clean = sanitize(raw_text)
        result = self.scorer.score_one(clean, source_trust=source_trust)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(decision=result.decision, score=result.score, latency_ms=latency_ms)

    def predict_label(self, raw_text: str, source_trust: float = 0.0) -> int:
        """Binary label for evaluation: 1 if flagged or blocked, else 0."""
        r = self.run(raw_text, source_trust=source_trust)
        return 1 if r.decision in ("flag", "block") else 0
