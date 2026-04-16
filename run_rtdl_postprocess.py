# """
# run_rtdl_postprocess.py
# =======================
# Post-hoc 2-opt+RTDL refinement of tours produced by INViT.

# Loads a trained INViT checkpoint, runs greedy inference on random or
# file-based TSP instances, then applies 2-opt+RTDL post-processing and
# reports average tour lengths and wall-clock times.

# Usage (examples)
# ----------------
# # Evaluate on 256 random TSP-100 instances using GPU 0:
#     python run_rtdl_postprocess.py \\
#         --checkpoint ckpt/tsp/train/model/checkpoint_<stamp>-n100-gpu0.pkl \\
#         --nb_nodes 100 \\
#         --num_instances 256 \\
#         --gpu_id 0

# # Larger instances, slower RTDL (recompute every 10 improvements):
#     python run_rtdl_postprocess.py \\
#         --checkpoint ckpt/tsp/train/model/checkpoint_<stamp>-n1000-gpu0.pkl \\
#         --nb_nodes 1000 \\
#         --num_instances 64 \\
#         --rtdl_freq 10 \\
#         --max_iter 300 \\
#         --gpu_id 0

# # CPU only, save JSON results:
#     python run_rtdl_postprocess.py \\
#         --checkpoint path/to/ckpt.pkl \\
#         --nb_nodes 100 \\
#         --gpu_id -1 \\
#         --out_file results.json
# """

# import os
# import sys
# import time
# import json
# import argparse

# import numpy as np
# import torch

# # ---------------------------------------------------------------------------
# # Make sure the project root is on sys.path so local modules are importable
# # ---------------------------------------------------------------------------
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from TSP_net import TSP_net
# from utils.utils_for_model import (
#     generate_tsp_instance,
#     compute_tsp_tour_length,
# )
# from utils.rtdl_utils import (
#     batch_two_opt_rtdl,
#     compute_dist_matrix,
#     tour_length,
# )


# # ---------------------------------------------------------------------------
# # Argument parsing
# # ---------------------------------------------------------------------------

# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(
#         description="Post-hoc 2-opt+RTDL for INViT TSP",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     # ---- checkpoint --------------------------------------------------------
#     p.add_argument("--checkpoint", type=str, required=True,
#                    help="Path to .pkl checkpoint saved by training_loop.py")

#     # ---- INViT architecture (must match the checkpoint) --------------------
#     p.add_argument("--nb_nodes",                  type=int, default=100)
#     p.add_argument("--dim_input_nodes",           type=int, default=2,
#                    help="Input feature dim (2 = x,y coordinates)")
#     p.add_argument("--dim_emb",                   type=int, default=128)
#     p.add_argument("--dim_ff",                    type=int, default=512)
#     p.add_argument("--nb_heads",                  type=int, default=8)
#     p.add_argument("--num_state_encoder",         type=int, default=2)
#     p.add_argument("--nb_layers_state_encoder",   type=int, default=2)
#     p.add_argument("--nb_layers_action_encoder",  type=int, default=2)
#     p.add_argument("--nb_layers_decoder",         type=int, default=3)
#     p.add_argument("--action_k",                  type=int, default=15,
#                    help="k for the action (candidate) k-NN set")
#     p.add_argument("--batchnorm",                 action="store_true")
#     p.add_argument("--if_agg_whole_graph",        action="store_true")
#     p.add_argument("--if_use_local_mask",         action="store_true")

#     # ---- inference ---------------------------------------------------------
#     p.add_argument("--bsz",           type=int, default=64,
#                    help="Total batch size passed to generate_tsp_instance; "
#                         "must be divisible by aug_num")
#     p.add_argument("--aug_num",       type=int, default=16,
#                    help="Augmentations per problem instance (aug_num | bsz)")
#     p.add_argument("--test_aug_num",  type=int, default=16,
#                    help="Alias for aug_num used by generate_tsp_instance")
#     p.add_argument("--aug",           type=str, default="mix",
#                    choices=["mix", "x8", "none"])
#     p.add_argument("--num_instances", type=int, default=256,
#                    help="How many unique TSP instances to evaluate")

#     # ---- 2-opt+RTDL --------------------------------------------------------
#     p.add_argument("--rtdl_freq", type=int, default=5,
#                    help="Number of 2-opt improvements between RTDL recomputes "
#                         "(paper uses 5 for TSP-500, 100 for TSP-10000)")
#     p.add_argument("--max_iter",  type=int, default=500,
#                    help="Max 2-opt outer-loop iterations per instance")

#     # ---- hardware ----------------------------------------------------------
#     p.add_argument("--gpu_id", type=str, default="0",
#                    help="Visible GPU id; use -1 for CPU")

#     # ---- output ------------------------------------------------------------
#     p.add_argument("--out_file", type=str, default=None,
#                    help="Optional path to save JSON results")

#     return p


# # ---------------------------------------------------------------------------
# # Model helpers
# # ---------------------------------------------------------------------------

# def _infer_arch_from_state_dict(sd: dict, fallback: argparse.Namespace) -> dict:
#     """
#     Inspect a raw state_dict and return the architecture hyper-parameters
#     needed to instantiate TSP_net correctly.

#     All values are derived from tensor shapes so the script never mis-matches
#     the checkpoint, regardless of the --nb_layers_* / --num_state_encoder
#     flags the user passed.

#     Parameters inferred
#     -------------------
#     num_state_encoder        : number of entries in state_encoders ModuleList
#     nb_layers_state_encoder  : MHA layers per state encoder
#     nb_layers_action_encoder : MHA layers in action encoder
#     nb_layers_decoder        : from WK_att_decoder output shape
#     dim_emb                  : from any weight whose dim is a multiple of 128
#     dim_ff                   : from the first FF linear layer in any encoder
#     dim_input_nodes          : from state_encoder input_emb weight shape
#     """
#     keys = list(sd.keys())

#     # ---- num_state_encoder: count distinct state_encoders.X.* indices ----
#     import re
#     se_indices = set(
#         int(m.group(1))
#         for k in keys
#         for m in [re.match(r"state_encoders\.(\d+)\.", k)]
#         if m
#     )
#     num_state_encoder = max(se_indices) + 1 if se_indices else fallback.num_state_encoder

#     # ---- dim_emb: from WK_att_decoder output width / nb_layers_decoder ----
#     # WK_att_decoder shape: (nb_layers_decoder*dim_emb, (num_state_encoder+1)*dim_emb)
#     wk_key = next((k for k in keys if "WK_att_decoder.weight" in k), None)
#     if wk_key:
#         out_dim, in_dim = sd[wk_key].shape          # out=(nld*de), in=(nse+1)*de
#         # dim_emb divides both; gcd gives the base unit
#         from math import gcd
#         de = gcd(out_dim, in_dim)
#         # Refine: de must also divide (num_state_encoder+1)*de
#         # in_dim = (num_state_encoder+1)*dim_emb  → dim_emb = in_dim/(nse+1)
#         dim_emb = in_dim // (num_state_encoder + 1)
#         nb_layers_decoder = out_dim // dim_emb
#     else:
#         dim_emb           = fallback.dim_emb
#         nb_layers_decoder = fallback.nb_layers_decoder

#     # ---- nb_layers_state_encoder: count MHA_layers per state_encoder -----
#     se_mha = set(
#         int(m.group(1))
#         for k in keys
#         for m in [re.match(r"state_encoders\.0\.encoder\.MHA_layers\.(\d+)\.", k)]
#         if m
#     )
#     nb_layers_state_encoder = (max(se_mha) + 1) if se_mha else fallback.nb_layers_state_encoder

#     # ---- nb_layers_action_encoder: count MHA_layers in action_encoder ----
#     ae_mha = set(
#         int(m.group(1))
#         for k in keys
#         for m in [re.match(r"action_encoder\.encoder\.MHA_layers\.(\d+)\.", k)]
#         if m
#     )
#     nb_layers_action_encoder = (max(ae_mha) + 1) if ae_mha else fallback.nb_layers_action_encoder

#     # ---- dim_ff: from first linear1 in any encoder -----------------------
#     ff_key = next(
#         (k for k in keys if "encoder.linear1_layers.0.weight" in k), None
#     )
#     dim_ff = sd[ff_key].shape[0] if ff_key else fallback.dim_ff

#     # ---- dim_input_nodes: from input_emb weight --------------------------
#     inp_key = next(
#         (k for k in keys if "input_emb.weight" in k), None
#     )
#     dim_input_nodes = sd[inp_key].shape[1] if inp_key else fallback.dim_input_nodes

#     arch = dict(
#         dim_input_nodes          = dim_input_nodes,
#         dim_emb                  = dim_emb,
#         dim_ff                   = dim_ff,
#         num_state_encoder        = num_state_encoder,
#         nb_layers_state_encoder  = nb_layers_state_encoder,
#         nb_layers_action_encoder = nb_layers_action_encoder,
#         nb_layers_decoder        = nb_layers_decoder,
#         nb_heads                 = fallback.nb_heads,   # not easily inferred
#         batchnorm                = fallback.batchnorm,
#         if_agg_whole_graph       = fallback.if_agg_whole_graph,
#     )
#     return arch


# def load_model(args, device: torch.device) -> TSP_net:
#     """
#     Load a TSP_net from checkpoint, auto-detecting all architecture params
#     from the state_dict so the model always matches the saved weights.

#     The checkpoint was saved by training_loop.py as:
#         {
#             'model_baseline': state_dict,
#             'model_train':    state_dict,
#             'optimizer':      ...,
#         }
#     We prefer 'model_baseline' (the EMA / greedy-best weights used at test time).
#     """
#     ckpt = torch.load(args.checkpoint, map_location=device)

#     # Prefer 'model_baseline'; fall back to 'model_train' if absent
#     state_dict = ckpt.get("model_baseline") or ckpt.get("model_train")
#     if state_dict is None:
#         raise KeyError(
#             "Checkpoint has neither 'model_baseline' nor 'model_train' key. "
#             f"Found keys: {list(ckpt.keys())}"
#         )

#     # Auto-detect architecture from checkpoint weights
#     arch = _infer_arch_from_state_dict(state_dict, fallback=args)

#     print(f"[load] checkpoint  : {args.checkpoint}")
#     print(f"[load] architecture detected from checkpoint:")
#     for k, v in arch.items():
#         cli_val = getattr(args, k, "—")
#         flag    = "  ← overrides CLI" if v != cli_val else ""
#         print(f"         {k:30s} = {v}{flag}")

#     # Update args so the rest of the script uses the correct state_k size
#     args.num_state_encoder       = arch["num_state_encoder"]
#     args.nb_layers_state_encoder = arch["nb_layers_state_encoder"]
#     args.nb_layers_action_encoder= arch["nb_layers_action_encoder"]
#     args.nb_layers_decoder       = arch["nb_layers_decoder"]
#     args.dim_emb                 = arch["dim_emb"]
#     args.dim_ff                  = arch["dim_ff"]
#     args.dim_input_nodes         = arch["dim_input_nodes"]
#     # Resize state_k to match actual num_state_encoder
#     args.state_k = [35, 50, 65][: args.num_state_encoder]

#     model = TSP_net(
#         arch["dim_input_nodes"],
#         arch["dim_emb"],
#         arch["dim_ff"],
#         arch["num_state_encoder"],
#         arch["nb_layers_state_encoder"],
#         arch["nb_layers_action_encoder"],
#         arch["nb_layers_decoder"],
#         arch["nb_heads"],
#         batchnorm=arch["batchnorm"],
#         if_agg_whole_graph=arch["if_agg_whole_graph"],
#     )

#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()

#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"[load] parameters  : {n_params:,}")
#     return model


# # ---------------------------------------------------------------------------
# # Inference helpers
# # ---------------------------------------------------------------------------

# def invit_greedy(
#     model:             TSP_net,
#     x_aug:             torch.Tensor,
#     action_k:          int,
#     state_k:           list,
#     if_use_local_mask: bool = False,
# ) -> torch.Tensor:
#     """
#     Run INViT in deterministic (greedy) mode.

#     Args:
#         x_aug:  (bsz, n, 2) augmented instance tensor on the correct device.

#     Returns:
#         tour tensor of shape (bsz, n) with integer node indices.
#     """
#     with torch.no_grad():
#         tour, _ = model(
#             x_aug, action_k, state_k,
#             choice_deterministic=True,
#             if_use_local_mask=if_use_local_mask,
#         )
#     return tour  # (bsz, n)


# def pick_best_augmentation(
#     tours:   torch.Tensor,
#     coords:  torch.Tensor,
#     aug_num: int,
# ) -> tuple:
#     """
#     From `aug_num` augmented tours per instance, return the shortest one.

#     Args:
#         tours:   (bsz, n) integer tour tensor;  bsz = n_instances * aug_num.
#         coords:  (bsz, n, 2) coordinate tensor; same ordering as tours.
#         aug_num: number of augmented copies per instance.

#     Returns:
#         best_tours  : (n_instances, n)   – best tour per instance
#         best_coords : (n_instances, n, 2)– coordinates of the first
#                        (un-augmented) copy; distances are invariant under
#                        rotation / reflection so any copy is equivalent.
#         best_lengths: (n_instances,)     – length of the selected tour
#     """
#     bsz          = tours.shape[0]
#     n_instances  = bsz // aug_num

#     # Compute tour length for every augmented tour
#     lengths = compute_tsp_tour_length(coords, tours)   # (bsz,)

#     # Reshape to (n_instances, aug_num), pick best per instance
#     lengths_mat  = lengths.view(n_instances, aug_num)
#     best_aug_idx = lengths_mat.argmin(dim=1)           # (n_instances,)

#     # Absolute flat indices into the (bsz,) dimension
#     offsets   = torch.arange(n_instances, device=tours.device) * aug_num
#     flat_idx  = offsets + best_aug_idx                 # (n_instances,)

#     best_tours   = tours[flat_idx]                     # (n_instances, n)
#     best_lengths = lengths_mat.min(dim=1).values       # (n_instances,)

#     # Use un-augmented coordinates (first copy of each instance)
#     # Distances are invariant under rotation/reflection, so this is correct.
#     first_idx   = offsets                              # index 0 of each group
#     best_coords = coords[first_idx]                    # (n_instances, n, 2)

#     return best_tours, best_coords, best_lengths


# # ---------------------------------------------------------------------------
# # Main evaluation loop
# # ---------------------------------------------------------------------------

# def evaluate(args, model: TSP_net, device: torch.device) -> dict:
#     """
#     Run INViT inference, then 2-opt+RTDL, and compare results.

#     Returns a results dictionary.
#     """
#     action_k = args.action_k
#     state_k  = args.state_k   # e.g. [35, 50, 65][:num_state_encoder]

#     instances_per_batch = args.bsz // args.aug_num
#     n_batches           = max(1, args.num_instances // instances_per_batch)
#     total_instances     = n_batches * instances_per_batch

#     print(f"\n{'=' * 60}")
#     print(f" INViT + 2-opt+RTDL  |  n={args.nb_nodes}")
#     print(f"{'=' * 60}")
#     print(f" batches        : {n_batches} × {instances_per_batch} instances")
#     print(f" augmentations  : {args.aug_num}  (aug='{args.aug}')")
#     print(f" total instances: {total_instances}")
#     print(f" RTDL freq      : {args.rtdl_freq}  |  max_iter: {args.max_iter}")
#     print(f"{'=' * 60}\n")

#     all_invit  = []
#     all_rtdl   = []
#     t_invit    = 0.0
#     t_rtdl     = 0.0

#     for batch_idx in range(n_batches):

#         # ---- generate instances --------------------------------------------
#         # generate_tsp_instance returns:
#         #   x_aug    : (bsz, n, 2)  – augmented coordinates
#         #   x_repeat : (bsz, n, 2)  – original coordinates repeated aug_num×
#         x_aug, x_repeat = generate_tsp_instance(args, device, if_test=True)

#         # ---- INViT greedy inference ----------------------------------------
#         t0 = time.time()
#         tours = invit_greedy(
#             model, x_aug, action_k, state_k,
#             if_use_local_mask=args.if_use_local_mask,
#         )
#         t_invit += time.time() - t0

#         # Select best augmentation per instance
#         best_tours, best_coords, invit_lengths = pick_best_augmentation(
#             tours, x_repeat, args.aug_num,
#         )

#         # Convert to numpy for RTDL processing
#         tours_np  = best_tours.cpu().numpy().astype(np.int32)   # (ni, n)
#         coords_np = best_coords.cpu().numpy()                   # (ni, n, 2)
#         invit_l   = invit_lengths.cpu().numpy()                 # (ni,)
#         all_invit.extend(invit_l.tolist())

#         # ---- 2-opt + RTDL --------------------------------------------------
#         t0 = time.time()
#         _, rtdl_lens, _ = batch_two_opt_rtdl(
#             tours=tours_np,
#             coords_batch=coords_np,
#             rtdl_update_freq=args.rtdl_freq,
#             max_iterations=args.max_iter,
#             verbose=True,
#         )
#         t_rtdl += time.time() - t0
#         all_rtdl.extend(rtdl_lens)

#         # ---- per-batch summary ---------------------------------------------
#         mean_i = float(np.mean(invit_l))
#         mean_r = float(np.mean(rtdl_lens))
#         delta  = (mean_i - mean_r) / mean_i * 100.0
#         print(
#             f"  Batch {batch_idx + 1:3d}/{n_batches}  |  "
#             f"INViT={mean_i:.4f}  RTDL={mean_r:.4f}  Δ={delta:+.2f}%"
#         )

#     # ---- overall summary ---------------------------------------------------
#     mean_invit = float(np.mean(all_invit))
#     mean_rtdl  = float(np.mean(all_rtdl))
#     delta_pct  = (mean_invit - mean_rtdl) / mean_invit * 100.0

#     print(f"\n{'=' * 60}")
#     print(f" RESULTS  (n={args.nb_nodes}, {total_instances} instances)")
#     print(f"{'=' * 60}")
#     print(f" INViT avg length   : {mean_invit:.5f}  [{t_invit:.1f}s total]")
#     print(f" 2-opt+RTDL avg len : {mean_rtdl:.5f}  [{t_rtdl:.1f}s total]")
#     print(f" Improvement        : {delta_pct:+.3f}%")
#     print(f"{'=' * 60}\n")

#     return {
#         "nb_nodes":         args.nb_nodes,
#         "num_instances":    total_instances,
#         "rtdl_freq":        args.rtdl_freq,
#         "max_iter":         args.max_iter,
#         "mean_invit":       mean_invit,
#         "mean_rtdl":        mean_rtdl,
#         "improvement_pct":  round(delta_pct, 4),
#         "time_invit_s":     round(t_invit, 2),
#         "time_rtdl_s":      round(t_rtdl, 2),
#         "all_invit_lengths": all_invit,
#         "all_rtdl_lengths":  all_rtdl,
#     }


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------

# def main():
#     parser = build_parser()
#     args   = parser.parse_args()

#     # Synchronise aug_num / test_aug_num (both read by generate_tsp_instance)
#     args.test_aug_num = args.aug_num

#     # Validate batch size divisibility
#     if args.bsz % args.aug_num != 0:
#         parser.error(
#             f"--bsz ({args.bsz}) must be divisible by --aug_num ({args.aug_num})"
#         )

#     # state_k mirrors the training config (first num_state_encoder values)
#     args.state_k = [35, 50, 65][: args.num_state_encoder]

#     # ---- hardware ----------------------------------------------------------
#     if args.gpu_id == "-1":
#         device = torch.device("cpu")
#         print("[hw] using CPU")
#     else:
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if device.type == "cuda":
#             print(f"[hw] GPU {args.gpu_id}: {torch.cuda.get_device_name(0)}")
#         else:
#             print("[hw] CUDA not available, falling back to CPU")

#     # ---- load model --------------------------------------------------------
#     model = load_model(args, device)

#     # ---- run ---------------------------------------------------------------
#     results = evaluate(args, model, device)

#     # ---- save results ------------------------------------------------------
#     if args.out_file:
#         os.makedirs(os.path.dirname(os.path.abspath(args.out_file)),
#                     exist_ok=True)
#         with open(args.out_file, "w") as f:
#             json.dump(results, f, indent=2)
#         print(f"[out] results saved to: {args.out_file}")


# if __name__ == "__main__":
#     main()

"""
run_rtdl_postprocess.py
=======================
Unified evaluation script for INViT with two optional RTDL enhancements:

  1. RTDL Inference Probability Adjustment (--use_rtdl_inference)
     Adjusts the decoder's output distribution at every autoregressive step
     by penalising edges that cross MST cluster boundaries.

  2. 2-opt + RTDL Post-processing  (on by default, disable with --no_postprocess)
     Topology-guided local search applied after tour construction.

Both flags are independent and can be combined freely, giving four modes:

  Mode A  Plain INViT                  (neither flag)
  Mode B  INViT + inference adj.       (--use_rtdl_inference)
  Mode C  INViT + post-proc            (default, no extra flag needed)
  Mode D  INViT + inference + post-proc  (--use_rtdl_inference)

The script always runs the plain-INViT baseline alongside whichever RTDL
options are active so you always get a direct comparison.

Usage
-----
# Plain baseline + post-processing (Mode A vs C):
    python run_rtdl_postprocess.py \\
        --checkpoint ckpt/tsp/train/model/checkpoint_<stamp>-n100-gpu0.pkl \\
        --nb_nodes 100 --num_instances 256 --gpu_id 0

# Add inference adjustment (Mode A vs B vs C vs D):
    python run_rtdl_postprocess.py \\
        --checkpoint path/to/ckpt.pkl \\
        --nb_nodes 100 --use_rtdl_inference --rtdl_mu 0.5 --gpu_id 0

# Inference adjustment only, no post-processing (Mode A vs B):
    python run_rtdl_postprocess.py \\
        --checkpoint path/to/ckpt.pkl \\
        --nb_nodes 100 --use_rtdl_inference --rtdl_mu 0.5 --no_postprocess

# Large instances — tune frequency and mu:
    python run_rtdl_postprocess.py \\
        --checkpoint path/to/ckpt.pkl \\
        --nb_nodes 1000 --use_rtdl_inference --rtdl_mu 0.3 \\
        --rtdl_freq 10 --max_iter 300 --gpu_id 0

# Save results to JSON:
    python run_rtdl_postprocess.py \\
        --checkpoint path/to/ckpt.pkl \\
        --nb_nodes 100 --use_rtdl_inference \\
        --gpu_id -1 --out_file results/run1.json
"""

import os
import sys
import re
import time
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# RTDLTSPNet is a drop-in TSP_net subclass; it accepts an optional
# rtdl_guide keyword argument in forward(). When the guide is None
# it is 100% identical to the original TSP_net.
from tsp_net_rtdl import RTDLTSPNet
from rtdl_inference import RTDLInferenceGuide
from rtdl_utils import batch_two_opt_rtdl, tour_length
from utils.utils_for_model import generate_tsp_instance, compute_tsp_tour_length


# ---------------------------------------------------------------------------
# Architecture auto-detection from checkpoint state_dict
# ---------------------------------------------------------------------------

def _infer_arch_from_state_dict(sd: dict, fallback: argparse.Namespace) -> dict:
    """
    Read all architecture hyper-parameters directly from checkpoint tensor
    shapes. This means the script never mis-matches a checkpoint regardless
    of what CLI flags were passed.

    Detected parameters
    -------------------
    num_state_encoder        — count of state_encoders.X.* index groups
    nb_layers_state_encoder  — MHA layer count inside state_encoders.0
    nb_layers_action_encoder — MHA layer count inside action_encoder
    nb_layers_decoder        — from WK_att_decoder output dimension
    dim_emb                  — from WK_att_decoder input dimension
    dim_ff                   — from first encoder linear1 layer
    dim_input_nodes          — from input_emb.weight input dimension
    """
    keys = list(sd.keys())

    se_indices = set(
        int(m.group(1))
        for k in keys
        for m in [re.match(r"state_encoders\.(\d+)\.", k)]
        if m
    )
    num_state_encoder = max(se_indices) + 1 if se_indices else fallback.num_state_encoder

    wk_key = next((k for k in keys if "WK_att_decoder.weight" in k), None)
    if wk_key:
        out_dim, in_dim   = sd[wk_key].shape
        dim_emb           = in_dim // (num_state_encoder + 1)
        nb_layers_decoder = out_dim // dim_emb
    else:
        dim_emb           = fallback.dim_emb
        nb_layers_decoder = fallback.nb_layers_decoder

    se_mha = set(
        int(m.group(1))
        for k in keys
        for m in [re.match(r"state_encoders\.0\.encoder\.MHA_layers\.(\d+)\.", k)]
        if m
    )
    nb_layers_state_encoder = (max(se_mha) + 1) if se_mha else fallback.nb_layers_state_encoder

    ae_mha = set(
        int(m.group(1))
        for k in keys
        for m in [re.match(r"action_encoder\.encoder\.MHA_layers\.(\d+)\.", k)]
        if m
    )
    nb_layers_action_encoder = (max(ae_mha) + 1) if ae_mha else fallback.nb_layers_action_encoder

    ff_key  = next((k for k in keys if "encoder.linear1_layers.0.weight" in k), None)
    inp_key = next((k for k in keys if "input_emb.weight" in k), None)

    return dict(
        dim_input_nodes          = sd[inp_key].shape[1] if inp_key else fallback.dim_input_nodes,
        dim_emb                  = dim_emb,
        dim_ff                   = sd[ff_key].shape[0] if ff_key else fallback.dim_ff,
        num_state_encoder        = num_state_encoder,
        nb_layers_state_encoder  = nb_layers_state_encoder,
        nb_layers_action_encoder = nb_layers_action_encoder,
        nb_layers_decoder        = nb_layers_decoder,
        nb_heads                 = fallback.nb_heads,
        batchnorm                = fallback.batchnorm,
        if_agg_whole_graph       = fallback.if_agg_whole_graph,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args, device: torch.device) -> RTDLTSPNet:
    """
    Load checkpoint as RTDLTSPNet, auto-detecting all architecture params.

    RTDLTSPNet is a subclass of TSP_net.  When called without rtdl_guide it
    is byte-for-byte identical to TSP_net.forward(), so no accuracy is lost
    compared to the original loader.

    Prefers 'model_baseline' weights (the EMA/greedy-best model saved by
    training_loop.py).
    """
    ckpt = torch.load(args.checkpoint, map_location=device)

    state_dict = ckpt.get("model_baseline") or ckpt.get("model_train")
    if state_dict is None:
        raise KeyError(
            "Checkpoint has neither 'model_baseline' nor 'model_train'. "
            f"Keys found: {list(ckpt.keys())}"
        )

    arch = _infer_arch_from_state_dict(state_dict, fallback=args)

    print(f"[load] checkpoint  : {args.checkpoint}")
    print(f"[load] architecture detected from checkpoint:")
    for k, v in arch.items():
        cli_val = getattr(args, k, "—")
        flag    = "  ← overrides CLI" if v != cli_val else ""
        print(f"         {k:30s} = {v}{flag}")

    # Propagate to args so the rest of the script uses correct values
    args.num_state_encoder       = arch["num_state_encoder"]
    args.nb_layers_state_encoder = arch["nb_layers_state_encoder"]
    args.nb_layers_action_encoder= arch["nb_layers_action_encoder"]
    args.nb_layers_decoder       = arch["nb_layers_decoder"]
    args.dim_emb                 = arch["dim_emb"]
    args.dim_ff                  = arch["dim_ff"]
    args.dim_input_nodes         = arch["dim_input_nodes"]
    args.state_k                 = [35, 50, 65][: args.num_state_encoder]

    model = RTDLTSPNet(
        arch["dim_input_nodes"], arch["dim_emb"], arch["dim_ff"],
        arch["num_state_encoder"],
        arch["nb_layers_state_encoder"], arch["nb_layers_action_encoder"],
        arch["nb_layers_decoder"], arch["nb_heads"],
        batchnorm        = arch["batchnorm"],
        if_agg_whole_graph = arch["if_agg_whole_graph"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] parameters  : {n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def invit_greedy(
    model:             RTDLTSPNet,
    x_aug:             torch.Tensor,
    action_k:          int,
    state_k:           list,
    if_use_local_mask: bool = False,
    rtdl_guide: RTDLInferenceGuide = None,
) -> torch.Tensor:
    """
    Run INViT in deterministic (greedy) mode.

    Args:
        x_aug:       (bsz, n, 2) augmented instance tensor.
        rtdl_guide:  RTDLInferenceGuide with .precompute() already called
                     for this batch.  None → standard INViT (no adjustment).

    Returns:
        tours: (bsz, n) integer node-index tensor.
    """
    with torch.no_grad():
        tours, _ = model(
            x_aug, action_k, state_k,
            choice_deterministic = True,
            if_use_local_mask    = if_use_local_mask,
            rtdl_guide           = rtdl_guide,   # None is silently ignored
        )
    return tours


def pick_best_augmentation(
    tours:   torch.Tensor,   # (bsz, n)
    coords:  torch.Tensor,   # (bsz, n, 2)
    aug_num: int,
) -> tuple:
    """
    From aug_num augmented copies per instance, keep the shortest tour.

    Returns:
        best_tours  : (n_instances, n)
        best_coords : (n_instances, n, 2)  — first (unaugmented) copy
        best_lengths: (n_instances,)
    """
    bsz         = tours.shape[0]
    n_instances = bsz // aug_num

    lengths     = compute_tsp_tour_length(coords, tours)        # (bsz,)
    lengths_mat = lengths.view(n_instances, aug_num)
    best_idx    = lengths_mat.argmin(dim=1)                     # (n_instances,)

    offsets     = torch.arange(n_instances, device=tours.device) * aug_num
    flat_idx    = offsets + best_idx

    best_tours   = tours[flat_idx]
    best_lengths = lengths_mat.min(dim=1).values
    best_coords  = coords[offsets]   # first copy per instance

    return best_tours, best_coords, best_lengths


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args, model: RTDLTSPNet, device: torch.device) -> dict:
    """
    Run all active evaluation conditions and print a comparison table.

    Conditions always run
    ---------------------
    A  Plain INViT (baseline)

    Conditions run when --use_rtdl_inference is set
    ------------------------------------------------
    B  INViT + RTDL inference probability adjustment

    Conditions run unless --no_postprocess is set
    ---------------------------------------------
    C  Plain INViT → 2-opt+RTDL post-processing
    D  INViT + RTDL inference → 2-opt+RTDL  (only when --use_rtdl_inference)
    """
    action_k         = args.action_k
    state_k          = args.state_k
    aug_num          = args.aug_num
    use_inf          = args.use_rtdl_inference
    do_post          = not args.no_postprocess

    instances_per_batch = args.bsz // aug_num
    n_batches           = max(1, args.num_instances // instances_per_batch)
    total_instances     = n_batches * instances_per_batch

    # ---- header ------------------------------------------------------------
    print(f"\n{'=' * 66}")
    print(f"  INViT RTDL Evaluation  |  n={args.nb_nodes}")
    print(f"{'=' * 66}")
    print(f"  instances          : {total_instances}  ({n_batches} × {instances_per_batch})")
    print(f"  augmentations      : {aug_num}  (aug='{args.aug}')")
    print(f"  inference adj.     : {'ON  (mu=' + str(args.rtdl_mu) + ')' if use_inf else 'OFF'}")
    print(f"  post-processing    : {'ON  (freq=' + str(args.rtdl_freq) + ', max_iter=' + str(args.max_iter) + ')' if do_post else 'OFF'}")
    print(f"{'=' * 66}\n")

    # Condition accumulators: lists of per-instance lengths
    acc  = {"A": [], "B": [], "C": [], "D": []}
    # Wall-clock time accumulators per phase
    t    = {"inv_plain": 0.0, "inv_adj": 0.0, "post_C": 0.0, "post_D": 0.0}

    for batch_idx in range(n_batches):

        # ---- generate instances --------------------------------------------
        x_aug, x_repeat = generate_tsp_instance(args, device, if_test=True)

        # ---- Condition A: plain INViT --------------------------------------
        t0       = time.time()
        tours_A  = invit_greedy(model, x_aug, action_k, state_k,
                                if_use_local_mask=args.if_use_local_mask)
        t["inv_plain"] += time.time() - t0

        best_A, coords_inst, lens_A = pick_best_augmentation(
            tours_A, x_repeat, aug_num
        )
        acc["A"].extend(lens_A.cpu().tolist())

        tours_np_A  = best_A.cpu().numpy().astype(np.int32)
        coords_np   = coords_inst.cpu().numpy()     # (n_inst, n, 2) — shared

        # ---- Condition B: INViT + RTDL inference adjustment ---------------
        if use_inf:
            guide = RTDLInferenceGuide(
                mu               = args.rtdl_mu,
                max_n_precompute = args.max_n_precompute,
            )
            # precompute uses the AUGMENTED batch (bsz, n, 2) to build MSTs
            # for all augmented copies; the guide is stateless per step.
            t0 = time.time()
            guide.precompute(x_aug.detach().cpu().numpy())
            tours_B = invit_greedy(model, x_aug, action_k, state_k,
                                   if_use_local_mask=args.if_use_local_mask,
                                   rtdl_guide=guide)
            t["inv_adj"] += time.time() - t0

            best_B, _, lens_B = pick_best_augmentation(tours_B, x_repeat, aug_num)
            acc["B"].extend(lens_B.cpu().tolist())
            tours_np_B = best_B.cpu().numpy().astype(np.int32)

        # ---- Condition C: plain INViT → 2-opt+RTDL ------------------------
        if do_post:
            t0 = time.time()
            _, lens_C, _ = batch_two_opt_rtdl(
                tours        = tours_np_A,
                coords_batch = coords_np,
                rtdl_update_freq = args.rtdl_freq,
                max_iterations   = args.max_iter,
                verbose          = False,
            )
            t["post_C"] += time.time() - t0
            acc["C"].extend(lens_C)

            # ---- Condition D: RTDL inference → 2-opt+RTDL ----------------
            if use_inf:
                t0 = time.time()
                _, lens_D, _ = batch_two_opt_rtdl(
                    tours        = tours_np_B,
                    coords_batch = coords_np,
                    rtdl_update_freq = args.rtdl_freq,
                    max_iterations   = args.max_iter,
                    verbose          = False,
                )
                t["post_D"] += time.time() - t0
                acc["D"].extend(lens_D)

        # ---- per-batch progress line --------------------------------------
        parts = [f"Batch {batch_idx + 1:3d}/{n_batches}"]
        parts.append(f"A={np.mean(acc['A'][-instances_per_batch:]):.4f}")
        if use_inf:
            parts.append(f"B={np.mean(acc['B'][-instances_per_batch:]):.4f}")
        if do_post:
            parts.append(f"C={np.mean(acc['C'][-instances_per_batch:]):.4f}")
        if do_post and use_inf:
            parts.append(f"D={np.mean(acc['D'][-instances_per_batch:]):.4f}")
        print("  " + "  |  ".join(parts))

    # ---- summary table -----------------------------------------------------
    ref = float(np.mean(acc["A"]))

    conditions = [
        ("A", "Plain INViT  (baseline)",               "inv_plain", None),
        ("B", "INViT + RTDL inference adj.",            "inv_adj",   None),
        ("C", "Plain INViT → 2-opt+RTDL",              "post_C",    "inv_plain"),
        ("D", "INViT + RTDL inf → 2-opt+RTDL",         "post_D",    "inv_adj"),
    ]

    print(f"\n{'=' * 66}")
    print(f"  RESULTS  (n={args.nb_nodes}, {total_instances} instances)")
    print(f"{'=' * 66}")
    print(f"  {'Condition':<40}  {'Avg Len':>9}  {'vs A':>7}  {'Time(s)':>8}")
    print(f"  {'-'*40}  {'-'*9}  {'-'*7}  {'-'*8}")

    results_out = {}
    for key, label, t_key, t_key2 in conditions:
        if not acc[key]:
            continue
        mean  = float(np.mean(acc[key]))
        delta = (ref - mean) / ref * 100.0
        # total time = inference time (this condition or shared) + post time
        wall  = t.get(t_key, 0.0) + (t.get(t_key2, 0.0) if t_key2 else 0.0)
        print(f"  {label:<40}  {mean:>9.5f}  {delta:>+6.2f}%  {wall:>8.1f}s")
        results_out[key] = {"mean": round(mean, 6), "vs_A_pct": round(delta, 4),
                            "time_s": round(wall, 2)}
    print(f"{'=' * 66}\n")

    return {
        "nb_nodes":           args.nb_nodes,
        "num_instances":      total_instances,
        "use_rtdl_inference": use_inf,
        "rtdl_mu":            args.rtdl_mu if use_inf else None,
        "postprocess":        do_post,
        "rtdl_freq":          args.rtdl_freq if do_post else None,
        "max_iter":           args.max_iter  if do_post else None,
        "conditions":         results_out,
        # per-instance arrays — useful for significance testing
        "lengths": {k: v for k, v in acc.items() if v},
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="INViT TSP evaluation with optional RTDL enhancements",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- checkpoint --------------------------------------------------------
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pkl checkpoint (architecture auto-detected)")

    # ---- architecture defaults (overridden by auto-detection) --------------
    arch = p.add_argument_group("Architecture (auto-detected from checkpoint)")
    arch.add_argument("--dim_input_nodes",           type=int, default=2)
    arch.add_argument("--dim_emb",                   type=int, default=128)
    arch.add_argument("--dim_ff",                    type=int, default=512)
    arch.add_argument("--nb_heads",                  type=int, default=8)
    arch.add_argument("--num_state_encoder",         type=int, default=2)
    arch.add_argument("--nb_layers_state_encoder",   type=int, default=2)
    arch.add_argument("--nb_layers_action_encoder",  type=int, default=2)
    arch.add_argument("--nb_layers_decoder",         type=int, default=3)
    arch.add_argument("--batchnorm",                 action="store_true")
    arch.add_argument("--if_agg_whole_graph",        action="store_true")
    arch.add_argument("--if_use_local_mask",         action="store_true")

    # ---- inference settings ------------------------------------------------
    inf = p.add_argument_group("Inference settings")
    inf.add_argument("--nb_nodes",       type=int, default=100)
    inf.add_argument("--num_instances",  type=int, default=256,
                     help="Total number of unique TSP instances to evaluate")
    inf.add_argument("--bsz",           type=int, default=64,
                     help="Batch size; must be divisible by --aug_num")
    inf.add_argument("--aug_num",       type=int, default=16,
                     help="Augmented copies per instance")
    inf.add_argument("--test_aug_num",  type=int, default=16)
    inf.add_argument("--aug",           type=str, default="mix",
                     choices=["mix", "x8", "none"])
    inf.add_argument("--action_k",      type=int, default=15)

    # ---- RTDL inference probability adjustment ----------------------------
    rtdl_inf = p.add_argument_group("RTDL inference probability adjustment")
    rtdl_inf.add_argument(
        "--use_rtdl_inference", action="store_true",
        help="Enable RTDL inference probability adjustment.  "
             "At each decoding step, candidate probabilities are re-weighted "
             "by their α-score (MST-cluster-boundary crossing penalty).  "
             "Adds conditions B and D to the output table."
    )
    rtdl_inf.add_argument(
        "--rtdl_mu", type=float, default=0.5,
        help="Penalty strength μ for inference adjustment.  "
             "0 = no adjustment, higher = more conservative about "
             "crossing MST cluster boundaries.  Typical range: 0.1 – 2.0."
    )
    rtdl_inf.add_argument(
        "--max_n_precompute", type=int, default=800,
        help="Instances with n ≤ this value get the full O(n²) bottleneck "
             "matrix precomputed once per batch (fast per-step lookup).  "
             "Larger instances use lazy O(n) DFS per step instead."
    )

    # ---- 2-opt + RTDL post-processing -------------------------------------
    post = p.add_argument_group("2-opt + RTDL post-processing")
    post.add_argument(
        "--no_postprocess", action="store_true",
        help="Disable 2-opt+RTDL post-processing entirely.  "
             "Only conditions A (and B if --use_rtdl_inference) are run."
    )
    post.add_argument(
        "--rtdl_freq", type=int, default=5,
        help="Successful 2-opt swaps between RTDL penalty recomputes.  "
             "Paper uses 5 for n=500, 100 for n=10000."
    )
    post.add_argument(
        "--max_iter", type=int, default=500,
        help="Hard cap on 2-opt outer-loop iterations per instance."
    )

    # ---- hardware / output ------------------------------------------------
    p.add_argument("--gpu_id",   type=str, default="0",
                   help="GPU id string; use -1 for CPU")
    p.add_argument("--out_file", type=str, default=None,
                   help="Optional path to write JSON results")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    args.test_aug_num = args.aug_num

    if args.bsz % args.aug_num != 0:
        parser.error(
            f"--bsz ({args.bsz}) must be divisible by --aug_num ({args.aug_num})"
        )

    # Placeholder; overwritten by _infer_arch_from_state_dict inside load_model
    args.state_k = [35, 50, 65][: args.num_state_encoder]

    # ---- hardware ----------------------------------------------------------
    if args.gpu_id == "-1":
        device = torch.device("cpu")
        print("[hw] using CPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"[hw] GPU {args.gpu_id}: {torch.cuda.get_device_name(0)}")
        else:
            print("[hw] CUDA not available, falling back to CPU")

    # ---- load model --------------------------------------------------------
    model = load_model(args, device)

    # ---- run ---------------------------------------------------------------
    results = evaluate(args, model, device)

    # ---- save results ------------------------------------------------------
    if args.out_file:
        out_dir = os.path.dirname(os.path.abspath(args.out_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Drop per-instance length arrays to keep JSON small
        slim = {k: v for k, v in results.items() if k != "lengths"}
        with open(args.out_file, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"[out] results saved to: {args.out_file}")


if __name__ == "__main__":
    main()