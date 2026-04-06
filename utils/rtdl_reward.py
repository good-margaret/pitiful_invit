import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
 
 
def _mst_length(dist_matrix: np.ndarray) -> float:
    """MST weight of a complete weighted graph."""
    return float(minimum_spanning_tree(csr_matrix(dist_matrix)).sum())
 
 
def compute_batch_rtdl_gap(
    tours_np:  np.ndarray,   # (B, n)  int node indices
    coords_np: np.ndarray,   # (B, n, 2)  2-D coordinates
) -> np.ndarray:
    """
    Compute the RTDL total gap for each tour in a batch.
 
    gap(tour) = L_tour - w_emax - L_MST  >= 0  (Theorem 1)
 
    Args:
        tours_np:  integer array of shape (B, n).
        coords_np: float array of shape (B, n, 2).
 
    Returns:
        gaps: float array of shape (B,), all values >= 0.
    """
    B, n = tours_np.shape
    gaps = np.empty(B, dtype=np.float64)
 
    for i in range(B):
        coords = coords_np[i]                          # (n, 2)
        tour   = tours_np[i]                           # (n,)
 
        # --- pairwise distance matrix  O(n^2) space ----------------------
        diff   = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dm     = np.sqrt((diff ** 2).sum(axis=-1))     # (n, n)
 
        # --- tour length and emax weight ---------------------------------
        edge_w  = np.array([dm[tour[t], tour[(t + 1) % n]] for t in range(n)])
        l_tour  = edge_w.sum()
        w_emax  = edge_w.max()
 
        # --- MST length  O(n^2 alpha(n)) ---------------------------------
        l_mst   = _mst_length(dm)
 
        # --- RTDL gap  (clamped to avoid tiny negative numerical noise) --
        gaps[i] = max(0.0, l_tour - w_emax - l_mst)
 
    return gaps
 