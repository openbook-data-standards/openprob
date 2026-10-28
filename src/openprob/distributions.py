from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isclose

from .contracts import probability


@dataclass(frozen=True, slots=True)
class CountDistribution:
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, tuple) or not 1 <= len(self.probabilities) <= 10001:
            raise ValueError("Supply an immutable probability vector for counts 0..N, N <= 10000")
        for value in self.probabilities:
            probability(value)
        if not isclose(fsum(self.probabilities), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Probabilities must sum to one; no implicit normalization")

    def at_least(self, threshold: int) -> float:
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
            raise ValueError("threshold must be a nonnegative integer")
        return min(1.0, fsum(self.probabilities[threshold:]))

    @property
    def mean(self) -> float:
        return fsum(index * value for index, value in enumerate(self.probabilities))


def empirical_counts(observations: tuple[int, ...], maximum: int) -> CountDistribution:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= 10000:
        raise ValueError("maximum must be an integer in [0, 10000]")
    if not isinstance(observations, tuple) or not 1 <= len(observations) <= 100000:
        raise ValueError("Supply 1..100000 immutable observations")
    counts = [0] * (maximum + 1)
    for value in observations:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            raise ValueError("Observation falls outside the declared support")
        counts[value] += 1
    return CountDistribution(tuple(value / len(observations) for value in counts))
