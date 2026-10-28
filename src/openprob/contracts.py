from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite


def text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be nonempty text of at most 4096 characters")


def timestamp(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamps must be timezone-aware datetimes")


def probability(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Probability must be a finite number in [0, 1]")
    if not 0 <= value <= 1 or not isfinite(value):
        raise ValueError("Probability must be a finite number in [0, 1]")


@dataclass(frozen=True, slots=True)
class Target:
    target_id: str
    event_id: str
    proposition: str
    settlement_basis: str
    outcome_at: datetime

    def __post_init__(self) -> None:
        for name in ("target_id", "event_id", "proposition", "settlement_basis"):
            text(getattr(self, name), name)
        timestamp(self.outcome_at)


@dataclass(frozen=True, slots=True)
class Forecast:
    forecast_id: str
    model_id: str
    model_version: str
    target: Target
    generated_at: datetime
    information_cutoff: datetime
    training_cutoff: datetime
    probability: float | None
    abstention_reason: str | None = None
    probability_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for name in ("forecast_id", "model_id", "model_version"):
            text(getattr(self, name), name)
        if not isinstance(self.target, Target):
            raise ValueError("target must be a Target")
        for value in (self.generated_at, self.information_cutoff, self.training_cutoff):
            timestamp(value)
        if not self.training_cutoff <= self.information_cutoff <= self.generated_at:
            raise ValueError("Require training_cutoff <= information_cutoff <= generated_at")
        if self.generated_at >= self.target.outcome_at:
            raise ValueError("Forecast must precede the target outcome time")
        if self.probability is None:
            text(self.abstention_reason, "abstention_reason")
            if self.probability_bounds is not None:
                raise ValueError("Abstentions cannot carry probability bounds")
        else:
            probability(self.probability)
            if self.abstention_reason is not None:
                raise ValueError("Scored forecasts cannot have an abstention reason")
            if self.probability_bounds is not None:
                if (
                    not isinstance(self.probability_bounds, tuple)
                    or len(self.probability_bounds) != 2
                ):
                    raise ValueError("probability_bounds must be an immutable pair")
                low, high = self.probability_bounds
                probability(low)
                probability(high)
                if not low <= self.probability <= high:
                    raise ValueError("Probability must lie within its bounds")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["target"]["outcome_at"] = self.target.outcome_at.isoformat()
        for name in ("generated_at", "information_cutoff", "training_cutoff"):
            result[name] = getattr(self, name).isoformat()
        return result


@dataclass(frozen=True, slots=True)
class Label:
    target: Target
    outcome: bool
    available_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.target, Target) or not isinstance(self.outcome, bool):
            raise ValueError("Label requires a Target and a boolean outcome")
        timestamp(self.available_at)
        if self.available_at < self.target.outcome_at:
            raise ValueError("Label cannot be available before the target outcome time")
