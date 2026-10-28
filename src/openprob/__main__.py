import json
from datetime import UTC, datetime, timedelta

from . import Forecast, Label, Target, empirical_counts, evaluate


def main() -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    distribution = empirical_counts((0, 1, 1, 2, 2, 2, 3, 3, 4, 5), maximum=5)
    forecasts, labels = [], []
    for threshold in (1, 2, 3):
        target = Target(
            f"synthetic:count:{threshold}",
            "synthetic:event",
            f"Fictional participant records at least {threshold} units",
            "Synthetic count, full fictional event, no overtime or voids",
            cutoff + timedelta(days=1),
        )
        forecasts.append(
            Forecast(
                f"synthetic:forecast:{threshold}",
                "empirical-counts",
                "0.1.0",
                target,
                cutoff,
                cutoff,
                cutoff,
                distribution.at_least(threshold),
            )
        )
        labels.append(Label(target, 2 >= threshold, cutoff + timedelta(days=2)))
    report = evaluate(tuple(forecasts), tuple(labels), cutoff + timedelta(days=3))
    print(
        json.dumps(
            {
                "synthetic": True,
                "warning": "Illustration only; correlated thresholds are not independent evaluation samples",
                "forecasts": [forecast.to_dict() for forecast in forecasts],
                "evaluation": report.to_dict(),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
