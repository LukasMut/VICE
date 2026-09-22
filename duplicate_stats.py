#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detect triplets that occur more often in the data than chance would predict.

Repeated trials are not by themselves a problem: if ``N`` triplets are drawn at
random from ``M`` objects, the birthday paradox guarantees a predictable number
of repeats. A clean uniform sample of 200,000 triplets over the 1,854 THINGS
objects contains around 38 repeated rows. Warning about those is noise.

What is worth a warning is a number of repeats that the sampling design cannot
explain -- a file that was concatenated with itself, say. So this module tests
the observed count against an explicit null instead of thresholding it.

Null model
----------
The ``N`` trials are drawn independently and uniformly from the
``K = C(M, 3)`` possible triplets. Trial identity is the *unordered* set
``{i, j, k}``: the same three objects shown twice is the same trial, whichever
column each object landed in. (``partition_triplets.py`` uses the same
convention.) The null is deliberately not taken over ordered rows, because the
odd-one-out choice is the signal VICE models, not a uniform draw over three
options -- two participants shown the same triplet agree far more often than
one time in three.

Test statistic
--------------
``W``, the number of *pairs* of trials that carry the same triplet. Writing
``I_ab`` for the indicator that trials ``a`` and ``b`` collide, the ``I_ab`` are
pairwise uncorrelated: for distinct ``a, b, c``,
``E[I_ab I_ac] = 1/K^2 = E[I_ab] E[I_ac]``. So ``W`` matches
``Binomial(C(N, 2), 1/K)`` in both moments exactly,

    E[W] = C(N, 2) / K          Var[W] = C(N, 2) (1/K) (1 - 1/K)

which is what makes a binomial reference distribution exactly right here rather
than merely convenient. The p-value is the one-sided upper tail. Only an excess
of repeats is flagged: a deficit means sampling without replacement, which is
what ``main_tripletize.py`` does on purpose.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from scipy.special import gammaln
from scipy.stats import binom

Tensor = torch.Tensor

# Warn only when the observed repeats fall in this upper tail under the null.
# Monkeypatch this rather than threading a keyword through load_data.
#
# Worth knowing where this bites: whenever fewer than -log(1 - 0.2) = 0.223
# collisions are expected, P(W >= 1) is already below the threshold and the
# test reduces exactly to "warn on any repeat at all". Sparse designs therefore
# keep the unconditional behaviour; only the dense regime, where repeats are
# genuinely predicted, goes quiet.
P_VALUE_THRESHOLD = 0.2

_LOG10 = math.log(10.0)


class DuplicateTripletsWarning(UserWarning):
    """Raised when the triplet data repeats triplets more often than chance."""


@dataclass(frozen=True)
class DuplicateReport:
    """Observed and expected repeat counts for one triplet file.

    Invariant: ``p_value == 0.0`` implies ``log10_p_value < -300``. The p-value
    underflows float64 long before the evidence stops being interpretable, so
    read ``log10_p_value`` rather than taking a log of ``p_value``.
    """

    file_name: str
    n_rows: int  # N
    n_objects: Optional[int]  # M
    n_cells: int  # K = C(M, 3)
    n_colliding_pairs: int  # W, the test statistic
    n_duplicated_rows: int  # rows sharing their triplet with another row
    n_duplicated_triplets: int  # distinct triplets that repeat
    n_duplicated_rows_exact: int  # identical *rows*, reported not tested
    n_degenerate_rows: int  # rows whose three indices are not distinct
    expected_colliding_pairs: float
    expected_duplicated_rows: float
    expected_duplicated_triplets: float
    p_value: float
    log10_p_value: float
    testable: bool  # False when the null is undefined for this input

    @property
    def is_significant(self) -> bool:
        """Whether this report warrants a warning."""
        if self.n_duplicated_rows == 0:
            return False
        # An undefined null must not silence the check: fall back to warning
        # about any repeat at all, without the statistical claim.
        if not self.testable:
            return True
        return self.p_value < P_VALUE_THRESHOLD

    def summary(self) -> str:
        """The warning body."""
        head = (
            f"\n...Found {self.n_duplicated_rows} of {self.n_rows} rows in "
            f"'{self.file_name}' that repeat one of {self.n_duplicated_triplets} "
            "distinct triplets.\n"
        )
        if self.testable:
            head += (
                f"...Chance alone predicts {self.expected_duplicated_rows:.1f} such "
                f"rows for M = {self.n_objects} objects and N = {self.n_rows} trials "
                f"(K = {self.n_cells:,} possible triplets): "
                f"{self.n_colliding_pairs} colliding pairs observed vs "
                f"{self.expected_colliding_pairs:.1f} expected, "
                f"{_format_pvalue(self.p_value, self.log10_p_value)}.\n"
            )
        else:
            head += (
                "...The expected number of repeats could not be computed for "
                f"M = {self.n_objects}, so this is a plain count with no test "
                "behind it.\n"
            )
        if self.n_degenerate_rows:
            head += (
                f"...{self.n_degenerate_rows} rows do not contain three distinct "
                "objects; those rows cannot arise under the null and are counted "
                "but not modelled.\n"
            )
        head += (
            "...Repeated trials are expected if the same triplet was shown to "
            "several participants; a rate far above chance usually means a file "
            "was concatenated twice. VICE trains on the data exactly as loaded.\n"
        )
        return head


def _format_pvalue(p_value: float, log10_p_value: float) -> str:
    if math.isnan(p_value):
        return "p could not be computed"
    if p_value >= 1e-300:
        return f"p = {p_value:.2g} under a binomial null"
    return f"p < 1e-300 (log10 p = {log10_p_value:.1f}) under a binomial null"


def n_cells(n_objects: int) -> int:
    """Number of distinct triplets that can be drawn from n_objects items."""
    if n_objects is None or n_objects < 3:
        return 0
    return math.comb(n_objects, 3)


def count_duplicates(triplets: Tensor) -> Tuple[int, int]:
    """Count rows that are *exactly* equal to another row.

    Column order is significant here, so ``[0, 1, 2]`` and ``[2, 1, 0]`` are
    different rows. Returns the number of rows belonging to a duplicated group
    and the number of distinct rows those collapse into. Reported for the
    user's benefit; the statistical test works on unordered triplets instead.
    """
    _, counts = torch.unique(triplets, dim=0, return_counts=True)
    repeated = counts[counts > 1]
    return int(repeated.sum().item()), int(repeated.shape[0])


def collision_counts(triplets: Tensor) -> Tuple[int, int, int]:
    """Count repeats of the unordered triplet {i, j, k}.

    Returns ``(n_colliding_pairs, n_duplicated_rows, n_duplicated_triplets)``.
    Sorting each row before the unique pass is the whole difference between
    row identity and trial identity.
    """
    if triplets.shape[0] == 0:
        return 0, 0, 0
    sorted_rows = torch.sort(triplets.long(), dim=1).values
    _, counts = torch.unique(sorted_rows, dim=0, return_counts=True)
    repeated = counts[counts > 1]
    n_colliding_pairs = int((counts * (counts - 1) // 2).sum().item())
    return n_colliding_pairs, int(repeated.sum().item()), int(repeated.shape[0])


def count_degenerate_rows(triplets: Tensor) -> int:
    """Count rows whose three indices are not distinct, e.g. [5, 5, 7]."""
    if triplets.shape[0] == 0:
        return 0
    a, b, c = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    return int(((a == b) | (a == c) | (b == c)).sum().item())


def expected_duplicates(n_rows: int, n_cells: int) -> Tuple[float, float, float]:
    """Expected repeat counts under the null, in numerically stable form.

    Returns ``(colliding_pairs, duplicated_rows, duplicated_triplets)``.

    The closed forms below are algebraically equal to the textbook occupancy
    expressions but are written with ``log1p``/``expm1`` throughout. This is not
    cosmetic: the direct form of the third quantity,
    ``K (1 - q**N - N p q**(N-1))``, subtracts numbers that agree to 16 digits
    when ``N**2 << K``. At ``M = 20000, N = 50`` it returns 5.28e-05 against a
    true value of 9.19e-10, and at ``M = 5000, N = 100`` it returns a negative
    count.

    Some cancellation survives in the third quantity even so: at the sparse
    extreme it holds about six significant digits rather than the full sixteen.
    That is far more than a warning message needs, and the p-value -- the only
    number the warning actually decides on -- does not depend on it.
    """
    N, K = n_rows, n_cells
    if N < 2 or K <= 0:
        return 0.0, 0.0, 0.0
    q = math.log1p(-1.0 / K)
    expected_pairs = math.comb(N, 2) / K
    expected_rows = -N * math.expm1((N - 1) * q)
    expected_triplets = -K * math.expm1((N - 1) * q + math.log1p((N - 1) / K))
    return expected_pairs, expected_rows, expected_triplets


def collision_pvalue(
    n_colliding_pairs: int, n_rows: int, n_cells: int
) -> Tuple[float, float]:
    """One-sided upper-tail probability of the observed collision count.

    Returns ``(p_value, log10_p_value)``. ``p_value`` underflows to 0.0 past
    roughly 1e-308; ``log10_p_value`` stays finite well beyond that, because the
    interesting cases (a file concatenated with itself) sit thousands of orders
    of magnitude into the tail and "p = 0.0" tells the user nothing.
    """
    N, K, w = n_rows, n_cells, n_colliding_pairs
    # No collisions is never surprising, whatever the null looks like -- and
    # this is also the N < 2 case, where there are no pairs to collide.
    if w <= 0:
        return 1.0, 0.0
    if N < 2 or K <= 0:
        return float("nan"), float("nan")
    n_pairs = math.comb(N, 2)
    p_value = float(binom.sf(w - 1, n_pairs, 1.0 / K))
    if p_value > 0.0:
        return p_value, math.log10(p_value)
    # binom.sf has underflowed, and binom.logsf/poisson.logsf return -inf here
    # rather than the log we need. Fall back to the Poisson tail in log space:
    # the leading term plus a geometric-ratio correction. Accurate to about
    # four decimals in log10 wherever the exact value is still computable.
    lam = n_pairs / K
    if w > lam > 0.0:
        log_tail = (-lam + w * math.log(lam) - float(gammaln(w + 1))) - math.log1p(
            -lam / w
        )
        return 0.0, log_tail / _LOG10
    return 0.0, float("-inf")


def analyse(
    triplets: Tensor, n_objects: Optional[int] = None, file_name: str = ""
) -> DuplicateReport:
    """Build a DuplicateReport for one triplet array."""
    testable = True

    if triplets.dim() != 2 or triplets.shape[1] != 3:
        # Not triplet-shaped; count nothing rather than silently comparing
        # scalars, but do not raise from what is only a data check.
        return DuplicateReport(
            file_name=file_name,
            n_rows=int(triplets.shape[0]) if triplets.dim() else 0,
            n_objects=n_objects,
            n_cells=0,
            n_colliding_pairs=0,
            n_duplicated_rows=0,
            n_duplicated_triplets=0,
            n_duplicated_rows_exact=0,
            n_degenerate_rows=0,
            expected_colliding_pairs=0.0,
            expected_duplicated_rows=0.0,
            expected_duplicated_triplets=0.0,
            p_value=float("nan"),
            log10_p_value=float("nan"),
            testable=False,
        )

    n_rows = int(triplets.shape[0])
    if n_rows and int(triplets.min().item()) < 0:
        # Negative indices are not object labels, so C(M, 3) is not the sample
        # space. Still report the repeats, just without the test.
        testable = False

    K = n_cells(n_objects) if n_objects is not None else 0
    if K <= 0:
        testable = False

    pairs, dup_rows, dup_triplets = collision_counts(triplets)
    dup_rows_exact, _ = count_duplicates(triplets) if n_rows else (0, 0)
    degenerate = count_degenerate_rows(triplets)

    if testable:
        exp_pairs, exp_rows, exp_triplets = expected_duplicates(n_rows, K)
        p_value, log10_p = collision_pvalue(pairs, n_rows, K)
    else:
        exp_pairs = exp_rows = exp_triplets = float("nan")
        p_value = log10_p = float("nan")

    return DuplicateReport(
        file_name=file_name,
        n_rows=n_rows,
        n_objects=n_objects,
        n_cells=K,
        n_colliding_pairs=pairs,
        n_duplicated_rows=dup_rows,
        n_duplicated_triplets=dup_triplets,
        n_duplicated_rows_exact=dup_rows_exact,
        n_degenerate_rows=degenerate,
        expected_colliding_pairs=exp_pairs,
        expected_duplicated_rows=exp_rows,
        expected_duplicated_triplets=exp_triplets,
        p_value=p_value,
        log10_p_value=log10_p,
        testable=testable,
    )


def warn_on_duplicates(
    triplets: Tensor, file_name: str, n_objects: Optional[int] = None
) -> DuplicateReport:
    """Warn if the data repeats triplets more often than chance would predict.

    Never alters or de-duplicates the data, and never raises: a data check must
    not be able to stop a training run.
    """
    report = analyse(triplets, n_objects=n_objects, file_name=file_name)
    if report.is_significant:
        warnings.warn(
            report.summary(),
            DuplicateTripletsWarning,
            stacklevel=3,
        )
    return report
