from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import fsum, log, log1p

from .contracts import Forecast, Label, timestamp


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    outcome_rate: float | None


@dataclass(frozen=True, slots=True)
class Evaluation:
    total: int
    abstained: int
    unresolved: int
    scored: int
    forecast_coverage: float
    brier_score: float | None
    mean_log_loss: float | None
    infinite_log_losses: int
    expected_calibration_error: float | None
    calibration: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(
    forecasts: tuple[Forecast, ...],
    labels: tuple[Label, ...],
    as_of: datetime,
    bins: int = 10,
) -> Evaluation:
    timestamp(as_of)
    if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 100:
        raise ValueError("bins must be in [1, 100]")
    if not isinstance(forecasts, tuple) or not isinstance(labels, tuple):
        raise ValueError("Supply immutable forecast and label tuples")
    if len(forecasts) > 100000 or len(labels) > 100000:
        raise ValueError("Evaluation is limited to 100000 forecasts and labels")
    if any(not isinstance(f, Forecast) for f in forecasts) or any(
        not isinstance(x, Label) for x in labels
    ):
        raise ValueError("Invalid forecast or label type")
    if len({f.forecast_id for f in forecasts}) != len(forecasts):
        raise ValueError("Duplicate forecast IDs")
    if len({f.target.target_id for f in forecasts}) != len(forecasts):
        raise ValueError("Select one forecast per target before evaluation")
    if len({(f.model_id, f.model_version) for f in forecasts}) > 1:
        raise ValueError("Evaluate one model version at a time")
    truth = {label.target.target_id: label for label in labels}
    if len(truth) != len(labels):
        raise ValueError("Duplicate labels")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    brier, losses = [], []
    abstained = unresolved = infinite = 0
    for forecast in forecasts:
        if forecast.generated_at > as_of:
            raise ValueError("Forecast generated after the evaluation cutoff")
        label = truth.get(forecast.target.target_id)
        if label is not None and label.target != forecast.target:
            raise ValueError("Label and forecast disagree on the target or settlement basis")
        if forecast.probability is None:
            abstained += 1
            continue
        if label is None or label.available_at > as_of:
            unresolved += 1
            continue
        p, y = forecast.probability, int(label.outcome)
        brier.append((p - y) ** 2)
        if (p == 0 and y == 1) or (p == 1 and y == 0):
            infinite += 1
        else:
            losses.append(-log(p) if y else -log1p(-p))
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    calibration = tuple(
        CalibrationBin(
            index / bins,
            (index + 1) / bins,
            len(bucket),
            fsum(p for p, _ in bucket) / len(bucket) if bucket else None,
            fsum(y for _, y in bucket) / len(bucket) if bucket else None,
        )
        for index, bucket in enumerate(buckets)
    )
    scored = len(brier)
    ece = (
        fsum(
            abs(item.mean_probability - item.outcome_rate) * item.count / scored
            for item in calibration
            if item.count and item.mean_probability is not None and item.outcome_rate is not None
        )
        if scored
        else None
    )
    return Evaluation(
        len(forecasts),
        abstained,
        unresolved,
        scored,
        (len(forecasts) - abstained) / len(forecasts) if forecasts else 0.0,
        fsum(brier) / scored if scored else None,
        fsum(losses) / scored if scored and not infinite else None,
        infinite,
        ece,
        calibration,
    )
