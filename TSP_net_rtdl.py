"""
tsp_net_rtdl.py
===============
RTDLTSPNet — a subclass of TSP_net that injects RTDL probability
adjustment into the autoregressive decoding loop.

The class is a drop-in replacement:
  • When rtdl_guide=None  : identical to TSP_net.forward() in every way.
  • When rtdl_guide is set: after the decoder outputs π_t(c), each
    candidate probability is re-weighted by the RTDL divergence signal
    and the distribution is renormalised before selection.

Usage
-----
    from tsp_net_rtdl import RTDLTSPNet
    from rtdl_inference import RTDLInferenceGuide

    model = RTDLTSPNet(...)
    model.load_state_dict(ckpt)
    model.eval()

    # Per-batch setup (before forward pass)
    guide = RTDLInferenceGuide(mu=0.5)
    guide.precompute(x_batch.cpu().numpy())

    with torch.no_grad():
        tours, log_p = model(
            x_batch, action_k, state_k,
            choice_deterministic=True,
            rtdl_guide=guide,          # ← new keyword arg
        )

    # Standard inference (no RTDL, identical to TSP_net)
    with torch.no_grad():
        tours, log_p = model(x_batch, action_k, state_k)
"""

import sys
import importlib
from typing import Optional, List

import torch
from torch.distributions import Categorical

from TSP_net import TSP_net
from rtdl_inference import RTDLInferenceGuide


# ---------------------------------------------------------------------------
# Extract module-level helpers from TSP_net's scope.
# These objects (knn, create_distance_mask_for_knn) live as module globals
# in TSP_net.py.  We fetch them so RTDLTSPNet.forward() can call them
# without modifying the original file.
# ---------------------------------------------------------------------------

def _import_tsp_helpers():
    """
    Return (knn_fn, create_mask_fn) from the TSP_net module.
    create_mask_fn may be None if not defined.
    """
    mod = sys.modules.get("TSP_net") or importlib.import_module("TSP_net")
    knn_fn       = getattr(mod, "knn", None)
    create_mask  = getattr(mod, "create_distance_mask_for_knn", None)
    if knn_fn is None:
        raise ImportError(
            "Could not find 'knn' in TSP_net module.  "
            "Make sure TSP_net.py is importable and defines 'knn'."
        )
    return knn_fn, create_mask


_knn, _create_distance_mask_for_knn = _import_tsp_helpers()


# ---------------------------------------------------------------------------
# RTDLTSPNet
# ---------------------------------------------------------------------------

class RTDLTSPNet(TSP_net):
    """
    TSP_net with optional RTDL-guided inference probability adjustment.

    Inherits all weights, architecture, and training behaviour from TSP_net.
    The only change is in forward(): one optional block that adjusts the
    decoder's output probabilities using RTDL divergence scores before the
    node selection step.

    At each decoding step t:
        1.  Decoder outputs  π_t(c)  for candidates c ∈ A^p   (unchanged)
        2.  If rtdl_guide is active:
              div(c)         = max(0, w(last, c) − bottleneck(last, c, MST))
              log π_adj(c)   = log π_t(c) − μ · div(c)
              π_adj          = softmax(log π_adj)            ← used for selection
        3.  Selection: argmax(π_adj) or sample from Categorical(π_adj)

    div(c) is the α-score of the candidate edge — zero for MST-consistent
    choices, large for edges that cross distant MST clusters.
    """

    def forward(
        self,
        x:                    torch.Tensor,
        action_k:             int,
        state_k:              List[int],
        choice_deterministic: bool = False,
        if_use_local_mask:    bool = False,
        rtdl_guide: Optional[RTDLInferenceGuide] = None,
    ):
        """
        Autoregressive tour construction with optional RTDL adjustment.

        Args (identical to TSP_net.forward unless noted)
        ---------------------------------------------------
        x                  : (bsz, nb_nodes, dim_input) coordinates
        action_k           : k for the action (candidate) k-NN set
        state_k            : list of k values for state encoder k-NN sets
        choice_deterministic: greedy (True) or stochastic (False)
        if_use_local_mask  : whether to use distance-based action mask
        rtdl_guide         : RTDLInferenceGuide with .precompute() already
                             called for this batch.  None → standard INViT.

        Returns
        -------
        tours               : (bsz, nb_nodes)  int   node visit order
        sumLogProbOfActions : (bsz,)           float sum of log-probs
        """
        # ------------------------------------------------------------------
        # Setup  (identical to TSP_net.forward)
        # ------------------------------------------------------------------
        bsz         = x.shape[0]
        nb_nodes    = x.shape[1]
        zero_to_bsz = torch.arange(bsz, device=x.device)

        # Random starting node for each instance in the batch
        start_idx   = torch.randint(nb_nodes, (bsz,), device=x.device)

        tours               = [start_idx]
        sumLogProbOfActions = []

        first_visited_node = x[zero_to_bsz, start_idx, :].view(bsz, 1, -1)
        last_visited_node  = first_visited_node.clone()

        # Current global node index — needed by RTDLInferenceGuide for
        # the bottleneck lookup.  Not present in the original TSP_net.
        current_node_idx = start_idx.clone()   # (bsz,)

        # Global unvisited mask and index table
        mask_global = torch.ones((bsz, nb_nodes), device=x.device, dtype=torch.bool)
        mask_global[zero_to_bsz, start_idx] = False
        all_idx = torch.arange(nb_nodes, device=x.device).unsqueeze(0).repeat(bsz, 1)

        # ------------------------------------------------------------------
        # Autoregressive decoding loop
        # ------------------------------------------------------------------
        for t in range(nb_nodes - 1):

            # ---- Unvisited node set -------------------------------------
            unvisited_matrix = all_idx[mask_global].view(bsz, -1)
            num_nodes        = unvisited_matrix.size(1)

            b_graph        = (torch.arange(bsz, device=x.device)
                              .repeat(num_nodes).sort()[0])
            unvisited_flat = unvisited_matrix.reshape(-1)
            graph          = x[b_graph, unvisited_flat].view(bsz, -1, self.dim_input)

            k_action = min(action_k, num_nodes)
            k_state  = (min(max(state_k), num_nodes)
                        if self.num_state_encoder > 0 else k_action)

            # ---- k-NN -------------------------------------------------
            knn_output = _knn(
                graph.view(-1, self.dim_input),
                last_visited_node.view(-1, self.dim_input),
                k_state, b_graph, zero_to_bsz,
            )
            knn_idx = (knn_output[1, :] % num_nodes).view(bsz, k_state).contiguous()

            # ---- Action encoder ----------------------------------------
            action_idx  = knn_idx[:, :k_action].contiguous()
            action_mask = None
            if if_use_local_mask and _create_distance_mask_for_knn is not None:
                action_mask = _create_distance_mask_for_knn(
                    last_visited_node, action_idx, graph
                )
            emb_action = self.action_encoder(
                graph, action_idx, last_visited_node, mask=action_mask
            )
            emb_q     = emb_action[:, k_action : k_action + 1, :]
            emb_other = emb_action[:, :k_action, :]

            # ---- State encoders ----------------------------------------
            for i in range(self.num_state_encoder):
                temp_k    = min(state_k[i], num_nodes)
                temp_idx  = knn_idx[:, :temp_k].contiguous()
                emb_state = self.state_encoders[i](
                    graph, temp_idx, last_visited_node, first_visited_node
                )
                emb_q     = torch.cat(
                    [emb_q, emb_state[:, temp_k     : temp_k + 1, :]], dim=2
                )
                emb_q     = torch.cat(
                    [emb_q, emb_state[:, temp_k + 1 : temp_k + 2, :]], dim=2
                )
                emb_other = torch.cat(
                    [emb_other, emb_state[:, :k_action, :]], dim=2
                )

            # ---- Map local knn indices → global node indices ------------
            # next_idx[b, j] = global index of the j-th action candidate
            # for instance b.  Needed both for RTDL lookup and for the
            # final last_visited_idx update.
            b_action     = (torch.arange(bsz, device=x.device)
                            .repeat(k_action).sort()[0])
            action_flat  = action_idx.reshape(-1)
            next_idx     = unvisited_matrix[b_action, action_flat].view(bsz, -1)
            # next_idx: (bsz, k_action)  — global node indices

            mask_for_decoder = action_mask.bool() if action_mask is not None else None

            # ---- Decoder -----------------------------------------------
            h_q           = self.query_mlp(emb_q)
            K_att_decoder = self.WK_att_decoder(emb_other)
            V_att_decoder = self.WV_att_decoder(emb_other)
            prob_next_node = self.decoder(
                h_q, K_att_decoder, V_att_decoder, mask_for_decoder
            )
            # prob_next_node: (bsz, k_action)

            # ==============================================================
            # RTDL INFERENCE PROBABILITY ADJUSTMENT
            # ==============================================================
            # This is the only block added relative to TSP_net.forward().
            # When rtdl_guide is None it is skipped entirely.
            if rtdl_guide is not None:
                # Candidate coordinates for edge-weight computation
                b_expand    = zero_to_bsz.unsqueeze(1).expand(bsz, k_action)
                cand_coords = x[b_expand, next_idx]   # (bsz, k_action, 2)

                prob_next_node = rtdl_guide.adjust_probs(
                    probs       = prob_next_node,
                    cur_idx     = current_node_idx,    # (bsz,) global idx
                    cand_idx    = next_idx,            # (bsz, k_action) global idx
                    last_coord  = last_visited_node,   # (bsz, 1, 2)
                    cand_coords = cand_coords,         # (bsz, k_action, 2)
                )
            # ==============================================================

            # ---- Node selection ----------------------------------------
            if choice_deterministic:
                idx = torch.argmax(prob_next_node, dim=1)
            else:
                idx = Categorical(prob_next_node).sample()

            # ---- State update ------------------------------------------
            last_visited_idx  = next_idx[zero_to_bsz, idx]            # (bsz,)
            last_visited_node = x[zero_to_bsz, last_visited_idx, :].view(bsz, 1, -1)
            current_node_idx  = last_visited_idx   # ← RTDL tracking

            ProbOfChoices = prob_next_node[zero_to_bsz, idx]
            sumLogProbOfActions.append(torch.log(ProbOfChoices))
            tours.append(last_visited_idx)

            mask_global[zero_to_bsz, last_visited_idx] = False

        # ------------------------------------------------------------------
        # Assemble outputs
        # ------------------------------------------------------------------
        sumLogProbOfActions = torch.stack(sumLogProbOfActions, dim=1).sum(dim=1)
        tours               = torch.stack(tours, dim=1)

        return tours, sumLogProbOfActions