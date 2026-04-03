"""
rtdl_utils.py
=============
RTD-Lite barcode computation and 2-opt+RTDL guided local search.

Reference:
    "Edge-wise Topological Divergence Gaps: Guiding Search in
     Combinatorial Optimization", Trofimov et al., arXiv:2512.15800, 2025

Correct algorithm (Theorem 1 from Trofimov et al.)
---------------------------------------------------
For each T_mst edge e = (u, v, w_mst):

    phi^{-1}(e) = the SMALLEST-WEIGHT T_path edge that makes u and v
                  connected when that edge is added together with ALL
                  lighter T_path edges AND all T_mst edges with weight
                  strictly less than w_mst.

Implementation (O(n^2) total):
    For each T_mst edge (in ascending weight order):
        1. Initialise a fresh Union-Find (uf) with T_mst edges < w_mst.
        2. Add T_path edges one by one (ascending) until u and v connect.
        3. The last T_path edge added = phi^{-1}(e).
        4. penalty = w(phi^{-1}) - w_mst  >= 0.

The penalties are indexed by TOUR-PATH edge (not MST edge) so that the
2-opt algorithm can directly rank tour edges by their "badness".

sum(penalties[e] for e in T_path) == L(s,t-tour) - L(T_mst)  (Theorem 1).
"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def compute_dist_matrix(coords):
    """
    Full pairwise Euclidean distance matrix.

    Args:
        coords: (n, 2) float array.
    Returns:
        (n, n) symmetric float64 matrix, diagonal = 0.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def tour_length(tour, dist_matrix):
    """Sum of edge weights for a closed Hamiltonian tour."""
    n = len(tour)
    return float(sum(
        dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n)
    ))


# ---------------------------------------------------------------------------
# MST
# ---------------------------------------------------------------------------

def _compute_mst_edges(dist_matrix):
    """
    MST of the complete graph as list of (weight, u, v) sorted ascending.
    """
    mst = minimum_spanning_tree(csr_matrix(dist_matrix))
    coo = mst.tocoo()
    return sorted(
        zip(coo.data.tolist(), coo.row.tolist(), coo.col.tolist()),
        key=lambda e: e[0],
    )


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Path-compressed, union-by-rank disjoint-set."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def copy(self):
        uf = _UnionFind(len(self.parent))
        uf.parent = list(self.parent)
        uf.rank   = list(self.rank)
        return uf


# ---------------------------------------------------------------------------
# RTDL barcode  (Theorem 1 from Trofimov et al. 2025)
# ---------------------------------------------------------------------------

def compute_rtdl_penalties(tour, dist_matrix):
    """
    Compute per-edge RTDL barcode penalties for a Hamiltonian tour.

    For each MST edge e = (u, v, w_mst):
        free set  = T_mst edges with weight < w_mst
        phi^{-1}(e) = lightest T_path edge completing u-v connectivity when
                      added on top of the free set + all lighter T_path edges
        penalty   = w(phi^{-1}(e)) - w_mst  >= 0

    Penalties are stored keyed by the T_path edge (not T_mst edge) so the
    2-opt algorithm can directly rank tour edges by their "badness".

    Complexity: O(n^2).  For large n consider skipping RTDL recompute and
    using cached penalties (controlled by rtdl_update_freq in two_opt_rtdl).

    Returns:
        dict { (min(u,v), max(u,v)): float } — one entry per tour edge,
        all values >= 0.
        The n-1 non-emax entries sum to L(s,t-tour) - L(MST)  (Theorem 1).
    """
    n  = len(tour)
    dm = np.asarray(dist_matrix, dtype=np.float64)

    # ---- A = T_path = tour minus emax  ------------------------------------
    all_edges = [
        (float(dm[tour[i], tour[(i + 1) % n]]),
         int(tour[i]),
         int(tour[(i + 1) % n]))
        for i in range(n)
    ]
    emax_idx  = max(range(n), key=lambda i: all_edges[i][0])
    emax_edge = all_edges[emax_idx]
    A_edges   = sorted(                        # ascending by weight
        [e for i, e in enumerate(all_edges) if i != emax_idx],
        key=lambda e: e[0],
    )

    # ---- T_mst  -----------------------------------------------------------
    B_edges = _compute_mst_edges(dm)           # already sorted ascending

    # ---- For each T_mst edge: fresh-UF + sequential T_path scan ----------
    penalties = {}

    for b_idx, (w_mst, eu, ev) in enumerate(B_edges):

        # Start: UF initialised with all T_mst edges lighter than w_mst
        uf = _UnionFind(n)
        for (w, u, v) in B_edges[:b_idx]:     # already sorted, stop early
            uf.union(u, v)

        # Add T_path edges in ascending order until eu and ev connect
        for (w_a, a, b) in A_edges:
            uf.union(a, b)
            if uf.find(eu) == uf.find(ev):
                key     = (min(a, b), max(a, b))
                penalty = max(0.0, w_a - w_mst)
                # Theorem 1 guarantees bijection: each T_path edge paired once.
                # In degenerate (tied) cases, keep the max penalty.
                penalties[key] = max(penalties.get(key, 0.0), penalty)
                break

    # ---- Ensure every T_path edge has an entry (0 if not the bottleneck) -
    for (w, a, b) in A_edges:
        key = (min(a, b), max(a, b))
        if key not in penalties:
            penalties[key] = 0.0

    # ---- emax: assign minimum positive penalty (empirically best) ---------
    pos_vals     = [p for p in penalties.values() if p > 1e-10]
    emax_penalty = min(pos_vals) if pos_vals else 0.0
    emax_key     = (min(emax_edge[1], emax_edge[2]),
                    max(emax_edge[1], emax_edge[2]))
    penalties[emax_key] = emax_penalty

    return penalties


# ---------------------------------------------------------------------------
# 2-opt + RTDL  (Algorithm 2 from Trofimov et al. 2025)
# ---------------------------------------------------------------------------

def two_opt_rtdl(tour, dist_matrix, max_iterations=1000, rtdl_update_freq=5):
    """
    2-opt local search with RTDL-guided candidate ordering.

    Differences from vanilla 2-opt:
      * Sort candidate edges by RTDL penalty (descending) — worst edge first.
      * Growing batch: start with top-10, expand by 10 when exhausted.
      * RTDL penalties refreshed every rtdl_update_freq successful swaps.

    Args:
        tour:             list of n node indices (Hamiltonian cycle, 0-indexed).
        dist_matrix:      (n, n) numpy distance matrix.
        max_iterations:   hard cap on outer loop.
        rtdl_update_freq: successful swaps between RTDL penalty recomputes.

    Returns:
        (improved_tour, final_length)
    """
    n    = len(tour)
    best = list(tour)
    dm   = np.asarray(dist_matrix, dtype=np.float64)

    best_len           = tour_length(best, dm)
    penalties          = compute_rtdl_penalties(best, dm)
    steps_since_update = 0

    def _penalty(i):
        u, v = best[i], best[(i + 1) % n]
        return penalties.get((min(u, v), max(u, v)), 0.0)

    N         = min(10, n)
    improved  = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved  = False
        iteration += 1

        order      = sorted(range(n), key=_penalty, reverse=True)
        candidates = order[:N]

        found = False
        for i in candidates:
            for j in candidates:
                pi, pj = (i, j) if i < j else (j, i)
                if pj - pi <= 1:
                    continue
                if pi == 0 and pj == n - 1:
                    continue   # degenerate reversal (same tour)

                a = best[pi];      b = best[pi + 1]
                c = best[pj];      d = best[(pj + 1) % n]

                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])

                if delta < -1e-10:
                    best[pi + 1:pj + 1] = best[pi + 1:pj + 1][::-1]
                    best_len += delta
                    steps_since_update += 1
                    found    = True
                    improved = True

                    if steps_since_update >= rtdl_update_freq:
                        penalties          = compute_rtdl_penalties(best, dm)
                        steps_since_update = 0
                    break

            if found:
                break

        if not found:
            if N < n:
                N        = min(N + 10, n)
                improved = True

    return best, float(best_len)


# ---------------------------------------------------------------------------
# Batch wrapper
# ---------------------------------------------------------------------------

def batch_two_opt_rtdl(tours, coords_batch, rtdl_update_freq=5,
                        max_iterations=1000, verbose=True):
    """
    Apply 2-opt+RTDL independently to each instance in a batch.

    Args:
        tours:            iterable of B tours (list/array of node indices).
        coords_batch:     (B, n, 2) node coordinates.
        rtdl_update_freq: successful swaps between RTDL recomputes.
        max_iterations:   per-instance 2-opt iteration cap.
        verbose:          print progress every ~10% of instances.

    Returns:
        (improved_tours, improved_lengths, initial_lengths)
    """
    B         = len(tours)
    log_every = max(1, B // 10)

    improved_tours, improved_lengths, initial_lengths = [], [], []

    for idx in range(B):
        if verbose and idx % log_every == 0:
            print(f"  [RTDL 2-opt] {idx + 1}/{B}")

        coords = np.asarray(coords_batch[idx], dtype=np.float64)
        tour   = [int(t) for t in tours[idx]]
        dm     = compute_dist_matrix(coords)

        init_l = tour_length(tour, dm)
        initial_lengths.append(init_l)

        imp_tour, imp_l = two_opt_rtdl(
            tour, dm,
            max_iterations=max_iterations,
            rtdl_update_freq=rtdl_update_freq,
        )
        improved_tours.append(imp_tour)
        improved_lengths.append(imp_l)

    return improved_tours, improved_lengths, initial_lengths