#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the duplicate-triplet statistics.

Everything here is deterministic and exact: the reference values come from
fractions.Fraction, not from Monte Carlo, so the suite stays fast and cannot
flake in CI.
"""

import math
import unittest
from collections import Counter
from fractions import Fraction

import duplicate_stats as ds
import torch
from scipy.stats import binom


def exact_expected_duplicated_triplets(n_rows: int, n_cells: int) -> float:
    """K (1 - q^N - N p q^(N-1)) in exact rational arithmetic."""
    p = Fraction(1, n_cells)
    q = 1 - p
    return float(n_cells * (1 - q ** n_rows - n_rows * p * q ** (n_rows - 1)))


def naive_expected_duplicated_triplets(n_rows: int, n_cells: int) -> float:
    """The textbook form, written the obvious way in floating point."""
    N, K = n_rows, n_cells
    q = math.log1p(-1.0 / K)
    return K * (1 - math.exp(N * q) - (N / K) * math.exp((N - 1) * q))


def exact_upper_tail(w: int, n: int, n_cells: int) -> float:
    """Exact binomial upper tail, for small n."""
    p = Fraction(1, n_cells)
    return float(sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(w, n + 1)))


class ExpectedCountsTestCase(unittest.TestCase):
    def test_expected_pairs_is_exact(self) -> None:
        for n_rows, n_objects in [(85, 20), (1500, 1854), (200000, 1854)]:
            K = ds.n_cells(n_objects)
            pairs, _, _ = ds.expected_duplicates(n_rows, K)
            self.assertEqual(pairs, math.comb(n_rows, 2) / K)

    def test_stable_form_matches_exact_rational(self) -> None:
        # Six significant digits at the sparse extreme, where some cancellation
        # survives even the stable form; far better elsewhere.
        for n_rows, n_objects in [(100, 1854), (1000, 1854), (50, 20000), (100, 5000)]:
            K = ds.n_cells(n_objects)
            _, _, stable = ds.expected_duplicates(n_rows, K)
            exact = exact_expected_duplicated_triplets(n_rows, K)
            self.assertLess(abs(stable / exact - 1.0), 1e-5)

    def test_naive_form_is_why_this_is_written_the_hard_way(self) -> None:
        """Lock in the stable form: the obvious algebra is catastrophically wrong.

        If someone 'simplifies' expected_duplicates back to the textbook
        expression, this test is what tells them why it was not written that
        way. At M=5000, N=100 the naive form returns a negative expected count;
        at M=20000, N=50 it is wrong by five orders of magnitude.
        """
        K = ds.n_cells(5000)
        self.assertLess(naive_expected_duplicated_triplets(100, K), 0.0)
        _, _, stable = ds.expected_duplicates(100, K)
        self.assertGreater(stable, 0.0)

        K = ds.n_cells(20000)
        exact = exact_expected_duplicated_triplets(50, K)
        naive = naive_expected_duplicated_triplets(50, K)
        _, _, stable = ds.expected_duplicates(50, K)
        self.assertLess(abs(stable / exact - 1.0), 1e-5)
        self.assertGreater(naive / exact, 1e4)

    def test_stable_form_agrees_with_naive_where_naive_works(self) -> None:
        """The stable form is not merely different; it agrees when both are valid."""
        K = 100
        _, _, stable = ds.expected_duplicates(10, K)
        self.assertAlmostEqual(stable, naive_expected_duplicated_triplets(10, K), places=9)
        self.assertAlmostEqual(stable, exact_expected_duplicated_triplets(10, K), places=9)

    def test_expected_rows_is_about_twice_expected_pairs_when_sparse(self) -> None:
        # each collision contributes two rows when triple collisions are negligible
        K = ds.n_cells(1854)
        pairs, rows, _ = ds.expected_duplicates(1500, K)
        self.assertAlmostEqual(rows / (2 * pairs), 1.0, places=5)

    def test_degenerate_inputs_return_zero(self) -> None:
        self.assertEqual(ds.expected_duplicates(1, 100), (0.0, 0.0, 0.0))
        self.assertEqual(ds.expected_duplicates(0, 100), (0.0, 0.0, 0.0))
        self.assertEqual(ds.expected_duplicates(100, 0), (0.0, 0.0, 0.0))


class PValueTestCase(unittest.TestCase):
    def test_matches_exact_rational_tail(self) -> None:
        n_rows, n_objects, w = 10, 6, 3
        K = ds.n_cells(n_objects)
        p, _ = ds.collision_pvalue(w, n_rows, K)
        self.assertAlmostEqual(p, exact_upper_tail(w, math.comb(n_rows, 2), K), places=12)

    def test_zero_collisions_is_certain(self) -> None:
        p, log10p = ds.collision_pvalue(0, 1000, ds.n_cells(1854))
        self.assertEqual(p, 1.0)
        self.assertEqual(log10p, 0.0)

    def test_monotone_in_the_statistic(self) -> None:
        K = ds.n_cells(60)
        previous = 1.0
        for w in range(0, 12):
            p, _ = ds.collision_pvalue(w, 400, K)
            self.assertLessEqual(p, previous)
            previous = p

    def test_log10_stays_finite_where_the_p_value_underflows(self) -> None:
        K = ds.n_cells(1854)
        p, log10p = ds.collision_pvalue(200000, 400000, K)
        self.assertEqual(p, 0.0)  # scipy underflows here
        self.assertTrue(math.isfinite(log10p))
        self.assertLess(log10p, -300.0)

    def test_log10_agrees_with_scipy_where_both_compute(self) -> None:
        K = ds.n_cells(1854)
        n_rows = 1500000
        for w in [1200, 1500, 2121]:
            _, log10p = ds.collision_pvalue(w, n_rows, K)
            reference = math.log10(binom.sf(w - 1, math.comb(n_rows, 2), 1.0 / K))
            self.assertAlmostEqual(log10p, reference, places=2)

    def test_scipy_handles_a_realistic_number_of_pairs(self) -> None:
        """Guards the pinned scipy==1.6.* against the newer one used in dev."""
        K = ds.n_cells(1854)
        p, _ = ds.collision_pvalue(500, 1000000, K)
        self.assertTrue(math.isfinite(p))
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_undefined_null_is_not_a_number(self) -> None:
        p, log10p = ds.collision_pvalue(5, 100, 0)
        self.assertTrue(math.isnan(p))
        self.assertTrue(math.isnan(log10p))


class CollisionCountsTestCase(unittest.TestCase):
    def test_permuted_rows_are_the_same_trial(self) -> None:
        triplets = torch.tensor([[0, 1, 2], [2, 1, 0], [1, 0, 2]])
        pairs, rows, distinct = ds.collision_counts(triplets)
        self.assertEqual((pairs, rows, distinct), (3, 3, 1))

    def test_column_order_still_distinguishes_exact_rows(self) -> None:
        triplets = torch.tensor([[0, 1, 2], [2, 1, 0]])
        self.assertEqual(ds.count_duplicates(triplets), (0, 0))
        self.assertEqual(ds.collision_counts(triplets)[0], 1)

    def test_three_identical_rows_give_three_pairs(self) -> None:
        triplets = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.assertEqual(ds.collision_counts(triplets), (3, 3, 1))

    def test_matches_brute_force(self) -> None:
        torch.manual_seed(0)
        triplets = torch.randint(0, 8, (50, 3))
        counts = Counter(tuple(sorted(row)) for row in triplets.tolist())
        expected_pairs = sum(math.comb(c, 2) for c in counts.values())
        expected_rows = sum(c for c in counts.values() if c > 1)
        expected_distinct = sum(1 for c in counts.values() if c > 1)
        self.assertEqual(
            ds.collision_counts(triplets),
            (expected_pairs, expected_rows, expected_distinct),
        )

    def test_degenerate_rows_are_counted(self) -> None:
        triplets = torch.tensor([[5, 5, 7], [0, 1, 2], [3, 3, 3]])
        self.assertEqual(ds.count_degenerate_rows(triplets), 2)

    def test_empty_input(self) -> None:
        self.assertEqual(ds.collision_counts(torch.empty(0, 3, dtype=torch.long)), (0, 0, 0))


class ReportTestCase(unittest.TestCase):
    @staticmethod
    def clean_sample(n_rows: int, n_objects: int) -> torch.Tensor:
        """Distinct triplets, so the null is satisfied exactly."""
        triplets = []
        i = j = 0
        k = 2
        while len(triplets) < n_rows:
            i, j, k = (i, j, k + 1)
            if k >= n_objects:
                j, k = j + 1, j + 2
            if j >= n_objects - 1:
                i, j, k = i + 1, i + 2, i + 3
            if k < n_objects:
                triplets.append([i, j, k])
        return torch.tensor(triplets)

    def test_clean_data_is_not_significant(self) -> None:
        report = ds.analyse(self.clean_sample(500, 200), n_objects=200)
        self.assertEqual(report.n_colliding_pairs, 0)
        self.assertEqual(report.p_value, 1.0)
        self.assertFalse(report.is_significant)

    def test_self_concatenated_file_is_significant(self) -> None:
        sample = self.clean_sample(500, 200)
        report = ds.analyse(torch.cat((sample, sample), dim=0), n_objects=200)
        self.assertEqual(report.n_colliding_pairs, 500)
        self.assertTrue(report.is_significant)
        self.assertEqual(report.p_value, 0.0)
        self.assertTrue(math.isfinite(report.log10_p_value))
        self.assertIn("log10 p", report.summary())

    def test_dense_regime_is_silent(self) -> None:
        """Many repeats, but no more than chance predicts for M=6, N=200."""
        torch.manual_seed(0)
        cells = self.clean_sample(20, 6)
        sample = cells[torch.randint(0, cells.shape[0], (200,))]
        report = ds.analyse(sample, n_objects=6)
        self.assertGreater(report.n_duplicated_rows, 150)
        self.assertFalse(report.is_significant)

    def test_undefined_null_falls_back_to_an_unconditional_warning(self) -> None:
        triplets = torch.tensor([[0, 1, 2], [0, 1, 2]])
        report = ds.analyse(triplets, n_objects=2)  # K = 0
        self.assertFalse(report.testable)
        self.assertTrue(report.is_significant)
        self.assertIn("could not be computed", report.summary())

    def test_negative_indices_are_reported_but_not_tested(self) -> None:
        triplets = torch.tensor([[-1, 1, 2], [-1, 1, 2]])
        report = ds.analyse(triplets, n_objects=10)
        self.assertFalse(report.testable)
        self.assertTrue(report.is_significant)

    def test_missing_n_objects_is_not_testable(self) -> None:
        report = ds.analyse(torch.tensor([[0, 1, 2], [0, 1, 2]]), n_objects=None)
        self.assertFalse(report.testable)

    def test_single_row_and_empty_input(self) -> None:
        for triplets in [torch.tensor([[0, 1, 2]]), torch.empty(0, 3, dtype=torch.long)]:
            report = ds.analyse(triplets, n_objects=10)
            self.assertEqual(report.p_value, 1.0)
            self.assertFalse(report.is_significant)

    def test_wrong_shape_does_not_raise(self) -> None:
        report = ds.analyse(torch.tensor([0, 1, 2, 0, 1, 2]), n_objects=10)
        self.assertFalse(report.testable)
        self.assertFalse(report.is_significant)

    def test_a_few_extra_repeats_are_not_yet_surprising(self) -> None:
        """3 collisions where chance predicts 2.4 is not evidence of anything."""
        sample = self.clean_sample(400, 60)
        report = ds.analyse(torch.cat((sample, sample[:3]), dim=0), n_objects=60)
        self.assertEqual(report.n_colliding_pairs, 3)
        self.assertGreater(report.p_value, 0.2)
        self.assertFalse(report.is_significant)

    def test_threshold_is_overridable(self) -> None:
        sample = self.clean_sample(400, 60)
        report = ds.analyse(torch.cat((sample, sample[:10]), dim=0), n_objects=60)
        original = ds.P_VALUE_THRESHOLD
        try:
            ds.P_VALUE_THRESHOLD = 1e-12
            self.assertFalse(report.is_significant)
            ds.P_VALUE_THRESHOLD = 0.2
            self.assertTrue(report.is_significant)
        finally:
            ds.P_VALUE_THRESHOLD = original
