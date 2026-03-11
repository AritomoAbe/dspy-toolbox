from pydantic import BaseModel


class SNRResult(BaseModel):
    snr: float
    avg_variance: float


class MonteCarloResult(BaseModel):
    easy: int
    medium: int
    hard: int
    probs: list[float]
