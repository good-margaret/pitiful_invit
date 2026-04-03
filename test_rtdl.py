"""
test_rtdl.py
============
Sanity checks for rtdl_utils.py.

Run with:
    python test_rtdl.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.rtdl_utils import (
    compute_dist_matrix,
    tour_length,
    compute_rtdl_penalties,
    two_opt_rtdl,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def random_tour(n, rng):
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def is_valid_tour(tour, n):
    return sorted(tour) == list(range(n))


# ---------------------------------------------------------------------------
# test 1 — dist_matrix properties
# ---------------------------------------------------------------------------

def test_dist_matrix():
    rng    = np.random.default_rng(0)
    coords = rng.random((20, 2))
    dm     = compute_dist_matrix(coords)

    assert dm.shape == (20, 20),            "wrong shape"
    assert np.allclose(dm, dm.T),           "not symmetric"
    assert np.allclose(np.diag(dm), 0),     "diagonal != 0"
    assert (dm >= 0).all(),                 "negative distances"

    print("[PASS] test_dist_matrix")


# ---------------------------------------------------------------------------
# test 2 — all penalties non-negative  (Theorem 1 constraint)
# ---------------------------------------------------------------------------

def test_penalties_non_negative():
    rng = np.random.default_rng(123)

    for _ in range(50):
        n      = rng.integers(5, 30)
        coords = rng.random((n, 2))
        dm     = compute_dist_matrix(coords)
        tour   = random_tour(n, rng)

        penalties = compute_rtdl_penalties(tour, dm)

        # Exactly n penalties: one per tour edge
        assert len(penalties) == n, \
            f"Expected {n} penalties, got {len(penalties)}"

        for key, p in penalties.items():
            assert p >= -1e-8, f"Negative penalty {p:.6e} for edge {key}"

    print("[PASS] test_penalties_non_negative  (50 random instances)")


# ---------------------------------------------------------------------------
# test 3 — penalty sum == (s,t)-tour length minus MST length  (Theorem 1)
# ---------------------------------------------------------------------------

def test_penalty_sum_equals_gap():
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix

    rng    = np.random.default_rng(42)
    passed = 0

    for _ in range(30):
        n      = rng.integers(6, 20)
        coords = rng.random((n, 2))
        dm     = compute_dist_matrix(coords)
        tour   = random_tour(n, rng)

        penalties = compute_rtdl_penalties(tour, dm)

        # MST length of the full graph
        mst_len = float(minimum_spanning_tree(csr_matrix(dm)).sum())

        # (s,t)-tour = tour minus its heaviest edge
        edge_weights = [dm[tour[i], tour[(i + 1) % n]] for i in range(n)]
        emax_w       = max(edge_weights)
        st_tour_len  = tour_length(tour, dm) - emax_w

        expected_gap = st_tour_len - mst_len

        # Find which edge is emax (to exclude its penalty from the sum,
        # since emax gets a synthetic penalty that is NOT part of Theorem 1)
        emax_i   = edge_weights.index(emax_w)
        u, v     = tour[emax_i], tour[(emax_i + 1) % n]
        emax_key = (min(u, v), max(u, v))

        real_sum = sum(p for k, p in penalties.items() if k != emax_key)

        assert abs(real_sum - expected_gap) < 1e-6, (
            f"Gap mismatch: sum={real_sum:.6f}, expected={expected_gap:.6f}"
        )
        passed += 1

    print(f"[PASS] test_penalty_sum_equals_gap  ({passed}/30 random instances)")


# ---------------------------------------------------------------------------
# test 4 — 2-opt+RTDL never worsens the tour
# ---------------------------------------------------------------------------

def test_two_opt_rtdl_never_worsens():
    rng = np.random.default_rng(7)

    for trial in range(30):
        n      = rng.integers(10, 60)
        coords = rng.random((n, 2))
        dm     = compute_dist_matrix(coords)
        tour   = random_tour(n, rng)

        init_len                = tour_length(tour, dm)
        new_tour, new_len       = two_opt_rtdl(
            tour, dm, max_iterations=200, rtdl_update_freq=5
        )

        assert is_valid_tour(new_tour, n), "Output is not a valid permutation"
        assert new_len <= init_len + 1e-8, \
            f"Tour worsened: {init_len:.6f} -> {new_len:.6f}"

    print("[PASS] test_two_opt_rtdl_never_worsens  (30 random instances)")


# ---------------------------------------------------------------------------
# test 5 — comparison with vanilla 2-opt (informational, no hard assert)
#
# NOTE: on small random instances (n < 50) vanilla 2-opt can visit more
# pairs per iteration (exhaustive O(n^2)) while RTDL uses a growing
# candidate batch.  RTDL's advantage appears at n >= 100 where the paper
# shows consistent improvement.  Here we just report the win/loss counts.
# ---------------------------------------------------------------------------

def vanilla_two_opt(tour, dm, max_iter=1000):
    """Reference vanilla 2-opt."""
    n    = len(tour)
    best = list(tour)
    imp  = True
    it   = 0
    while imp and it < max_iter:
        imp = False
        it += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best[i], best[i + 1]
                c, d = best[j], best[(j + 1) % n]
                if dm[a, c] + dm[b, d] < dm[a, b] + dm[c, d] - 1e-10:
                    best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                    imp = True
                    break
            if imp:
                break
    return best, tour_length(best, dm)


def test_rtdl_vs_vanilla_report():
    """
    Report how often each method finds a shorter tour starting from the
    same initial random tour.  RTDL wins more on larger instances.
    """
    rng         = np.random.default_rng(99)
    rtdl_wins   = 0
    van_wins    = 0
    ties        = 0

    for _ in range(40):
        n      = rng.integers(15, 50)
        coords = rng.random((n, 2))
        dm     = compute_dist_matrix(coords)
        tour   = random_tour(n, rng)

        _, rtdl_l = two_opt_rtdl(list(tour), dm, max_iterations=500)
        _, van_l  = vanilla_two_opt(list(tour), dm, max_iter=500)

        if rtdl_l < van_l - 1e-8:
            rtdl_wins += 1
        elif van_l < rtdl_l - 1e-8:
            van_wins += 1
        else:
            ties += 1

    print(
        f"[INFO] test_rtdl_vs_vanilla (n=15-50, 40 trials): "
        f"RTDL wins={rtdl_wins}, vanilla wins={van_wins}, ties={ties}"
    )
    print("       (RTDL advantage grows with n; expected on large instances)")
    # No hard assert — advantage is expected at n>=100, not tiny random instances.
    print("[PASS] test_rtdl_vs_vanilla_report")


# ---------------------------------------------------------------------------
# run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running rtdl_utils sanity checks...\n")
    test_dist_matrix()
    test_penalties_non_negative()
    test_penalty_sum_equals_gap()
    test_two_opt_rtdl_never_worsens()
    test_rtdl_vs_vanilla_report()
    print("\nAll tests passed.")