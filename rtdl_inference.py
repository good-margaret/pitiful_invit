"""
rtdl_inference.py
=================
RTDL-Guided Inference Probability Adjustment for autoregressive TSP solvers.

What is it?
-----------
During INViT's autoregressive tour construction, at each step t the decoder
produces a probability distribution  π_t(c)  over a set of candidate next
nodes c ∈ A^p.  Ordinarily the policy samples (or argmax-selects) from π_t
directly.

This module adjusts π_t BEFORE the selection by penalising candidates that
would introduce a topologically "bad" edge — one that bridges two distant
clusters in the minimum spanning tree:

    div(c) = max( 0,  w(last, c)  −  bottleneck(last, c, MST) )

where
    w(last, c)              = Euclidean distance from the last visited node
                              to candidate c
    bottleneck(u, v, MST)   = maximum edge weight on the unique path u → v
                              in the full-graph MST

The bottleneck equals the "birth time" at which u and v become connected
when MST edges are added in ascending weight order.  If  w(last, c)  greatly
exceeds this threshold, the edge (last, c) crosses an MST cluster boundary
much later than the MST does — a topological mismatch.

Connection to the RTDL barcode (Proposition 1, Trofimov et al. 2025):
    For a tour edge e = (u, v):
        penalty(e) = w(e) − bottleneck(u, v, MST)   ≡   α-score(e)
    so  div(c)  is exactly the α-score of the edge that would be added
    if candidate c were chosen next.

Adjusted distribution:
    log π_adj(c) = log π(c) − μ · div(c)
    π_adj        = softmax( log π_adj )

When μ = 0, the output is identical to the original π_t.

Public API
----------
    RTDLInferenceGuide(mu, max_n_precompute)
        .precompute(coords_batch_np)          — call once per batch
        .adjust_probs(probs, cur_idx,
                      cand_idx, last_coord,
                      cand_coords)            — call once per decoding step
"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from typing import Optional
import torch


# ---------------------------------------------------------------------------
# MST helpers
# ---------------------------------------------------------------------------

def _build_mst_adj(dist_matrix: np.ndarray) -> list:
    """
    Build an undirected adjacency list for the MST of dist_matrix.

    Returns:
        adj: list of length n, where adj[u] = [(v, w), ...].
    """
    n   = len(dist_matrix)
    mst = minimum_spanning_tree(csr_matrix(dist_matrix))
    coo = mst.tocoo()
    adj = [[] for _ in range(n)]
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        adj[int(u)].append((int(v), float(w)))
        adj[int(v)].append((int(u), float(w)))
    return adj


# ---------------------------------------------------------------------------
# Bottleneck (max-weight path in a tree) helpers
# ---------------------------------------------------------------------------

def _single_source_bottleneck(adj: list, src: int, n: int) -> np.ndarray:
    """
    Compute the bottleneck distance from src to every other node.

    bottleneck(src, j) = max edge weight on the unique path src → j in MST.

    Uses iterative DFS (avoids Python recursion limit).  O(n) time.

    Returns:
        bn: float32 array of length n.  bn[src] = 0.
    """
    bn      = np.zeros(n, dtype=np.float32)
    visited = np.zeros(n, dtype=bool)
    stack   = [(src, -1, 0.0)]          # (node, parent, max_w_from_src)

    while stack:
        node, parent, cur_max = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        bn[node]      = cur_max
        for nb, ew in adj[node]:
            if not visited[nb]:
                stack.append((nb, node, max(cur_max, ew)))
    return bn


def _all_pairs_bottleneck(adj: list, n: int) -> np.ndarray:
    """
    Compute the full n×n bottleneck matrix for a tree.

    bottleneck[i, j] = max edge weight on path i → j in MST.
    O(n²) time and O(n²) space.

    Returns:
        bn: float32 array of shape (n, n), symmetric, zero diagonal.
    """
    bn = np.empty((n, n), dtype=np.float32)
    for src in range(n):
        bn[src] = _single_source_bottleneck(adj, src, n)
    return bn


# ---------------------------------------------------------------------------
# Euclidean distance matrix
# ---------------------------------------------------------------------------

def _euclidean_dm(coords: np.ndarray) -> np.ndarray:
    """
    Full pairwise Euclidean distance matrix for coords of shape (n, 2).
    Returns float32 (n, n).
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]   # (n,n,2)
    return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)        # (n,n)


# ---------------------------------------------------------------------------
# RTDLInferenceGuide
# ---------------------------------------------------------------------------

class RTDLInferenceGuide:
    """
    Provides step-wise RTDL probability adjustment for autoregressive
    tour construction.

    For each candidate c at decoding step t:

        div(c) = max( 0,  w(last, c)  −  bottleneck(last, c, MST) )

    Adjusted distribution:

        log π_adj(c) = log π(c) − μ · div(c)
        π_adj        = softmax( log π_adj )

    A candidate c with  div(c) = 0  is fully consistent with the MST
    topology (reaching c costs no more than the MST path already implies).
    A candidate with large div(c) is "crossing a cluster boundary" —
    exactly the kind of edge the RTDL barcode flags as suboptimal.

    Parameters
    ----------
    mu : float
        Penalty strength.  mu=0 → no change.  Typical range 0.1 – 2.0.
        Higher values make the policy more conservative about long edges.
    max_n_precompute : int
        Instances with n ≤ this threshold get the full O(n²) bottleneck
        matrix precomputed once before decoding (fast per-step lookups).
        Larger instances use O(n) per-step DFS to avoid the O(n²) memory.
    """

    def __init__(self, mu: float = 0.5, max_n_precompute: int = 800):
        self.mu               = mu
        self.max_n_precompute = max_n_precompute

        self._bsz:  Optional[int]         = None
        self._n:    Optional[int]         = None
        self._mode: Optional[str]         = None   # 'precomputed' | 'lazy'

        # Precomputed mode: (bsz, n, n) float32
        self._bottleneck: Optional[np.ndarray] = None
        # Lazy mode: list[adj] of length bsz
        self._mst_adj:    Optional[list]       = None

    # ------------------------------------------------------------------
    # Setup — must be called once per batch before the forward pass
    # ------------------------------------------------------------------

    def precompute(self, coords_batch: np.ndarray) -> None:
        """
        Build the bottleneck data structures for a batch of instances.

        Must be called once per batch BEFORE forward().

        For n ≤ max_n_precompute: computes the full (bsz, n, n) bottleneck
        matrix.  Per-step adjustment is then O(1) (array lookup).

        For n > max_n_precompute: stores MST adjacency lists.  Per-step
        adjustment costs O(n) per instance (one DFS from the current node).

        Args:
            coords_batch: float array of shape (bsz, n, 2).
        """
        bsz, n, _ = coords_batch.shape
        self._bsz  = bsz
        self._n    = n

        if n <= self.max_n_precompute:
            self._mode       = 'precomputed'
            self._mst_adj    = None
            self._bottleneck = np.empty((bsz, n, n), dtype=np.float32)
            for i in range(bsz):
                dm               = _euclidean_dm(coords_batch[i])
                adj              = _build_mst_adj(dm)
                self._bottleneck[i] = _all_pairs_bottleneck(adj, n)

        else:
            # For large n: store MST adj lists; compute bottleneck on demand
            self._mode       = 'lazy'
            self._bottleneck = None
            self._mst_adj    = []
            for i in range(bsz):
                dm  = _euclidean_dm(coords_batch[i])
                adj = _build_mst_adj(dm)
                self._mst_adj.append(adj)

    # ------------------------------------------------------------------
    # Per-step probability adjustment
    # ------------------------------------------------------------------

    def adjust_probs(
        self,
        probs:       torch.Tensor,   # (bsz, k_action)
        cur_idx:     torch.Tensor,   # (bsz,)           int   global node index
        cand_idx:    torch.Tensor,   # (bsz, k_action)  int   global node indices
        last_coord:  torch.Tensor,   # (bsz, 1, 2)      float last visited coord
        cand_coords: torch.Tensor,   # (bsz, k_action, 2) float candidate coords
    ) -> torch.Tensor:               # (bsz, k_action)  adjusted probabilities
        """
        Return RTDL-adjusted candidate probabilities.

        Computation (CPU-side for MST lookup, then back to device):
            edge_w[b,j] = ||last[b] − cand[b,j]||
            bn[b,j]     = bottleneck(cur[b], cand[b,j], MST_b)
            div[b,j]    = max(0, edge_w[b,j] − bn[b,j])
            log π_adj   = log π − μ · div
            π_adj       = softmax(log π_adj)
        """
        if self.mu == 0.0:
            return probs

        device = probs.device
        bsz, k = probs.shape

        # ---- edge weights (from coordinates, fast on CPU or GPU) ---------
        # last_coord: (bsz, 1, 2),  cand_coords: (bsz, k, 2)
        edge_w_np = (
            torch.norm(cand_coords - last_coord, dim=-1)
            .detach().cpu().numpy().astype(np.float32)
        )   # (bsz, k)

        cur_np  = cur_idx.detach().cpu().numpy().astype(np.int32)   # (bsz,)
        cand_np = cand_idx.detach().cpu().numpy().astype(np.int32)  # (bsz, k)

        # ---- bottleneck lookup -------------------------------------------
        bn_np = self._lookup_bottleneck(cur_np, cand_np)            # (bsz, k)

        # ---- divergence scores  (>= 0) ----------------------------------
        div_np = np.maximum(0.0, edge_w_np - bn_np, dtype=np.float32)
        div    = torch.tensor(div_np, dtype=probs.dtype, device=device)

        # ---- log-space adjustment with numerical stability ---------------
        log_p     = torch.log(probs.clamp(min=1e-10))
        log_p_adj = log_p - self.mu * div
        # Subtract row-max before softmax to prevent overflow
        log_p_adj = log_p_adj - log_p_adj.max(dim=-1, keepdim=True).values
        adj_probs = torch.softmax(log_p_adj, dim=-1)

        return adj_probs

    # ------------------------------------------------------------------
    # Internal bottleneck lookup (dispatches to precomputed or lazy)
    # ------------------------------------------------------------------

    def _lookup_bottleneck(
        self,
        cur_np:  np.ndarray,   # (bsz,)   int
        cand_np: np.ndarray,   # (bsz, k) int
    ) -> np.ndarray:           # (bsz, k) float32
        bsz, k = cand_np.shape

        if self._mode == 'precomputed':
            # Vectorised 3-D fancy indexing: O(bsz * k)
            b_idx = np.arange(bsz, dtype=np.int32)[:, np.newaxis]  # (bsz,1)
            return self._bottleneck[b_idx, cur_np[:, np.newaxis], cand_np]

        else:
            # Lazy: one O(n) DFS per instance from the current node
            result = np.empty((bsz, k), dtype=np.float32)
            for b in range(bsz):
                bn_row      = _single_source_bottleneck(
                    self._mst_adj[b], int(cur_np[b]), self._n
                )
                result[b]   = bn_row[cand_np[b]]
            return result