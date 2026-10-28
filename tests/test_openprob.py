import unittest
from datetime import UTC, datetime, timedelta
from math import log

from openprob import CountDistribution, Forecast, Label, Target, empirical_counts, evaluate


class OpenProbContracts(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.target = Target(
            "fake:target",
            "fake:event",
            "Fictional outcome",
            "Synthetic rules",
            self.now + timedelta(days=1),
        )

    def forecast(self, p=0.75, **overrides):
        values = dict(
            forecast_id="fake:forecast",
            model_id="baseline",
            model_version="1",
            target=self.target,
            generated_at=self.now,
            information_cutoff=self.now,
            training_cutoff=self.now,
            probability=p,
        )
        values.update(overrides)
        return Forecast(**values)

    def test_scores_and_calibration(self):
        result = evaluate(
            (self.forecast(),),
            (Label(self.target, True, self.now + timedelta(days=2)),),
            self.now + timedelta(days=3),
        )
        self.assertAlmostEqual(result.brier_score, 0.0625)
        self.assertAlmostEqual(result.mean_log_loss, -log(0.75))
        self.assertAlmostEqual(result.expected_calibration_error, 0.25)
        self.assertEqual(result.scored, 1)

    def test_no_clipping_infinite_log_loss(self):
        result = evaluate(
            (self.forecast(0),),
            (Label(self.target, True, self.now + timedelta(days=2)),),
            self.now + timedelta(days=3),
        )
        self.assertEqual(result.infinite_log_losses, 1)
        self.assertIsNone(result.mean_log_loss)
        self.assertEqual(result.brier_score, 1)

    def test_abstentions_unresolved_and_empty(self):
        result = evaluate((self.forecast(None, abstention_reason="missing data"),), (), self.now)
        self.assertEqual(result.abstained, 1)
        self.assertEqual(result.forecast_coverage, 0)
        self.assertEqual(evaluate((self.forecast(),), (), self.now).unresolved, 1)
        self.assertIsNone(evaluate((), (), self.now).brier_score)

    def test_future_labels_are_not_used(self):
        result = evaluate(
            (self.forecast(),),
            (Label(self.target, True, self.now + timedelta(days=2)),),
            self.now + timedelta(days=1),
        )
        self.assertEqual(result.scored, 0)
        self.assertEqual(result.unresolved, 1)

    def test_invalid_probabilities(self):
        for p in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(p=p), self.assertRaises(ValueError):
                self.forecast(p)

    def test_timestamp_and_leakage_guards(self):
        for changes in (
            {"generated_at": self.now.replace(tzinfo=None)},
            {"training_cutoff": self.now + timedelta(seconds=1)},
            {"information_cutoff": self.now + timedelta(seconds=1)},
            {"generated_at": self.target.outcome_at},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.forecast(**changes)
        with self.assertRaises(ValueError):
            Label(self.target, True, self.now)
        with self.assertRaises(ValueError):
            evaluate((self.forecast(),), (), self.now - timedelta(seconds=1))

    def test_abstention_and_bounds(self):
        with self.assertRaises(ValueError):
            self.forecast(None)
        with self.assertRaises(ValueError):
            self.forecast(0.75, probability_bounds=(0.1, 0.5))
        self.assertEqual(self.forecast(0.75, probability_bounds=(0.5, 0.9)).probability, 0.75)
        with self.assertRaises(ValueError):
            self.forecast(0.75, abstention_reason="contradictory")

    def test_duplicates_and_settlement_mismatch(self):
        f = self.forecast()
        label = Label(self.target, True, self.now + timedelta(days=2))
        for forecasts, labels in (((f, f), ()), ((f,), (label, label))):
            with self.assertRaises(ValueError):
                evaluate(forecasts, labels, self.now + timedelta(days=3))
        other = Target(
            "fake:target",
            "fake:event",
            "Fictional outcome",
            "Different rules",
            self.target.outcome_at,
        )
        with self.assertRaises(ValueError):
            evaluate(
                (f,),
                (Label(other, True, self.now + timedelta(days=2)),),
                self.now + timedelta(days=3),
            )

    def test_distribution_is_coherent_and_bounded(self):
        d = empirical_counts((0, 1, 1, 2), maximum=2)
        self.assertEqual(d.at_least(0), 1)
        self.assertEqual(d.at_least(1), 0.75)
        self.assertEqual(d.at_least(2), 0.25)
        self.assertEqual(d.at_least(3), 0)
        self.assertEqual(d.mean, 1)
        for values in ((0.2, 0.2), (float("nan"), 1), [0.5, 0.5]):
            with self.assertRaises(ValueError):
                CountDistribution(values)
        with self.assertRaises(ValueError):
            empirical_counts((3,), maximum=2)
        with self.assertRaises(ValueError):
            d.at_least(True)
