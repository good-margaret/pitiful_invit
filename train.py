# import torch
# import torch.nn as nn
# import time
# import argparse

# import os
# import datetime

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# from TSP_net import TSP_net
# from VRP_net import VRP_net
# from utils.utils_for_model import create_parser, read_from_logs
# from training_loop import train_model_with_knn
# from test_function import run_tsp_test_knn,run_tsplib_test_knn,run_vrp_test_knn, run_cvrplib_test_knn



# ###################
# # Hardware : CPU / GPU(s)
# ###################

# device = torch.device("cpu"); gpu_id = -1 # select CPU

# gpu_id = '0' # select a single GPU  
# #gpu_id = '2,3' # select multiple GPUs  
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
    
# print(device)



# ### parser ###


# config_dict = {
#     'aug': 'mix',
#     'bsz': 64,
#     'nb_nodes':50,
#     'model_lr': 2e-5,
#     'nb_batch_per_epoch': 300,
#     'data_path':'./',
#     'checkpoint_model': 'n',
#     'aug_num': 16,
#     'test_aug_num': 16,
#     'num_state_encoder': 2,
#     'dim_emb': 128,
#     'dim_ff':512,
#     'nb_heads': 8,
#     'action_k': 15,
#     'nb_layers_state_encoder': 2,
#     'nb_layers_action_encoder': 2,
#     'nb_layers_decoder': 3,
#     'nb_epochs': 400,
#     'problem': 'tsp',
#     'gamma': 0.99,
#     'dim_input_nodes': 2,
#     'batchnorm':False,
#     'gpu_id': 0,
#     'loss_type':'n',
#     'train_joint':'n',
#     'nb_batch_eval': 80,
#     'if_use_local_mask':False,
#     'if_agg_whole_graph':False,
#     'tol':1e-3,
# }

# state_k = [35,50,65]
# custom_parser, args = create_parser(config_dict)
# config = custom_parser.parse_args(namespace=args)
# if args.checkpoint_model != 'n':
#     read_from_logs(args)

# args.state_k = state_k[:args.num_state_encoder]

# if args.problem == 'cvrp' or args.problem == 'sdvrp':
#     args.CAPACITIES = {
#                 10: 20.,
#                 20: 30.,
#                 50: 40.,
#                 100: 50.
#             }

# print(args)


# if args.problem == 'tsp':

#     model_train = TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.num_state_encoder, 
#                 args.nb_layers_state_encoder, args.nb_layers_action_encoder, args.nb_layers_decoder, args.nb_heads, batchnorm = args.batchnorm, if_agg_whole_graph = args.if_agg_whole_graph)
#     model_baseline = TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.num_state_encoder, 
#                 args.nb_layers_state_encoder, args.nb_layers_action_encoder, args.nb_layers_decoder, args.nb_heads, batchnorm = args.batchnorm, if_agg_whole_graph = args.if_agg_whole_graph)
    
# elif args.problem == 'cvrp' or args.problem == 'sdvrp':

#     model_train = VRP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.num_state_encoder,
#                 args.nb_layers_state_encoder,args.nb_layers_action_encoder, args.nb_layers_decoder, args.nb_heads, batchnorm = args.batchnorm, if_agg_whole_graph = args.if_agg_whole_graph)
#     model_baseline = VRP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.num_state_encoder,
#                 args.nb_layers_state_encoder,args.nb_layers_action_encoder, args.nb_layers_decoder, args.nb_heads, batchnorm = args.batchnorm, if_agg_whole_graph = args.if_agg_whole_graph)

# else:

#     raise ValueError('Unsupported Problem Type')

# optimizer_model = torch.optim.AdamW( model_train.parameters() , lr = args.model_lr ) 
# scheduler_model = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_model, gamma=args.gamma)
# model_train = model_train.to(device)
# model_baseline = model_baseline.to(device)
# if args.checkpoint_model != 'n':
#     save_addr_model = args.data_path+'ckpt/'+args.problem+'/train/model/checkpoint_'
#     checkpoint_file_model = save_addr_model + args.checkpoint_model+'.pkl'
#     checkpoint_model = torch.load(checkpoint_file_model, map_location=device)
#     tot_time_ckpt_model = checkpoint_model['tot_time']
#     model_baseline.load_state_dict(checkpoint_model['model_baseline'])
#     model_train.load_state_dict(checkpoint_model['model_train'])
#     optimizer_model.load_state_dict(checkpoint_model['optimizer'])
# model_baseline.eval()

# print(args); print('')

# # Logs
# #os.system("mkdir logs")
# time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
# file_name = args.data_path+'ckpt/'+args.problem+'/train/logs'+'/'+time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
# file = open(file_name,"w",1) 
# file.write(time_stamp+'\n\n') 
# for arg in vars(args):
#     file.write(arg)
#     hyper_param_val="={}".format(getattr(args, arg))
#     file.write(hyper_param_val)
#     file.write('\n')
# file.write('\n\n') 
# plot_performance_train = []
# plot_performance_baseline = []
# all_strings = []
# epoch_ckpt = 0
# tot_time_ckpt = 0


# # # Uncomment these lines to re-start training with saved checkpoint

# ###################
# # Main training loop 
# ###################

# train_model_with_knn(args,model_train,model_baseline,optimizer_model,scheduler_model,device,file,time_stamp)    


# ## final evaluation part

# if args.problem == 'tsp':

#     sizes = [100,1000,5000,10000]
#     bszs = [64,32,16,8]
#     num_instance = [500,50,5,5]
#     distributions = ['uniform', 'clustered1', 'clustered2', 'explosion', 'implosion']
#     local_k = args.action_k
#     global_k = args.state_k
#     if_use_local_mask = False
#     data_path = args.data_path +'data/'
#     run_tsp_test_knn(local_k,global_k,args.aug,model_baseline,if_use_local_mask,sizes,bszs,data_path,device,file,distributions,num_instance=num_instance)
#     run_tsplib_test_knn(model_baseline,args.action_k,args.state_k)

# elif args.problem == 'cvrp':

#     capacity = 50
#     sizes = [50,500,5000]
#     bszs = [64,32,16]
#     num_instance = [500,50,5]
#     distributions = ['uniform', 'clustered1', 'clustered2', 'explosion', 'implosion']
#     local_k = args.action_k
#     global_k = args.state_k
#     if_use_local_mask = False
#     data_path = args.data_path +'data/'
#     run_vrp_test_knn(local_k,global_k,args.aug,model_baseline,if_use_local_mask,sizes,bszs,data_path,device,file,distributions,num_instance)
#     run_cvrplib_test_knn(model_baseline,args.action_k,args.state_k)

"""
train.py
=============
Main training script for INViT with optional RTDL reward shaping.
When rtdl_lambda=0, behaves like standard training.
"""

import os
import sys
import datetime
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TSP_net import TSP_net
from VRP_net import VRP_net
from training_loop import train_model_with_knn
from test_function import (
    run_tsp_test_knn,
    run_tsplib_test_knn,
    run_vrp_test_knn,
    run_cvrplib_test_knn,
)


def _infer_arch_from_state_dict(sd: dict, fallback) -> dict:
    import re
    from math import gcd

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

    ff_key = next((k for k in keys if "encoder.linear1_layers.0.weight" in k), None)
    dim_ff = sd[ff_key].shape[0] if ff_key else fallback.dim_ff

    inp_key = next((k for k in keys if "input_emb.weight" in k), None)
    dim_input_nodes = sd[inp_key].shape[1] if inp_key else fallback.dim_input_nodes

    return dict(
        dim_input_nodes          = dim_input_nodes,
        dim_emb                  = dim_emb,
        dim_ff                   = dim_ff,
        num_state_encoder        = num_state_encoder,
        nb_layers_state_encoder  = nb_layers_state_encoder,
        nb_layers_action_encoder = nb_layers_action_encoder,
        nb_layers_decoder        = nb_layers_decoder,
        nb_heads                 = fallback.nb_heads,
        batchnorm                = fallback.batchnorm,
        if_agg_whole_graph       = fallback.if_agg_whole_graph,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="INViT TSP/CVRP training with optional RTDL reward shaping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--problem",   type=str, default="tsp",
                   choices=["tsp", "cvrp", "sdvrp"])
    p.add_argument("--nb_nodes",  type=int, default=100)

    rtdl = p.add_argument_group("RTDL reward shaping")
    rtdl.add_argument("--rtdl_lambda", type=float, default=0.0,
                      help="Weight of RTDL gap penalty. 0 = disabled, "
                           "typical range for enabled: 0.05–0.5")

    ft = p.add_argument_group("Fine-tuning")
    ft.add_argument("--finetune",       type=str, default=None,
                    metavar="CHECKPOINT")
    ft.add_argument("--reset_optimizer", action="store_true")
    ft.add_argument("--reset_scheduler", action="store_true")

    arch = p.add_argument_group("Architecture (auto-detected when --finetune is set)")
    arch.add_argument("--dim_input_nodes",          type=int, default=2)
    arch.add_argument("--dim_emb",                  type=int, default=128)
    arch.add_argument("--dim_ff",                   type=int, default=512)
    arch.add_argument("--nb_heads",                 type=int, default=8)
    arch.add_argument("--num_state_encoder",        type=int, default=2)
    arch.add_argument("--nb_layers_state_encoder",  type=int, default=2)
    arch.add_argument("--nb_layers_action_encoder", type=int, default=2)
    arch.add_argument("--nb_layers_decoder",        type=int, default=3)
    arch.add_argument("--batchnorm",                action="store_true")
    arch.add_argument("--if_agg_whole_graph",       action="store_true")

    train = p.add_argument_group("Training")
    train.add_argument("--nb_epochs",          type=int,   default=400)
    train.add_argument("--nb_batch_per_epoch", type=int,   default=300)
    train.add_argument("--nb_batch_eval",      type=int,   default=80)
    train.add_argument("--bsz",                type=int,   default=64)
    train.add_argument("--model_lr",           type=float, default=2e-5)
    train.add_argument("--gamma",              type=float, default=0.99)
    train.add_argument("--tol",                type=float, default=1e-3)
    train.add_argument("--aug",                type=str,   default="mix",
                       choices=["mix", "x8", "none"])
    train.add_argument("--aug_num",            type=int,   default=16)
    train.add_argument("--test_aug_num",       type=int,   default=16)
    train.add_argument("--action_k",           type=int,   default=15)
    train.add_argument("--if_use_local_mask",  action="store_true")
    train.add_argument("--loss_type",          type=str,   default="n")
    train.add_argument("--train_joint",        type=str,   default="n")

    p.add_argument("--gpu_id",    type=str, default="0")
    p.add_argument("--data_path", type=str, default="./")

    return p


def _build_model(arch: dict, problem: str):
    common = (
        arch["dim_input_nodes"], arch["dim_emb"], arch["dim_ff"],
        arch["num_state_encoder"],
        arch["nb_layers_state_encoder"], arch["nb_layers_action_encoder"],
        arch["nb_layers_decoder"], arch["nb_heads"],
    )
    kwargs = dict(batchnorm=arch["batchnorm"],
                  if_agg_whole_graph=arch["if_agg_whole_graph"])

    if problem == "tsp":
        return TSP_net(*common, **kwargs)
    else:
        return VRP_net(*common, **kwargs)


def main():
    parser = build_parser()
    args   = parser.parse_args()

    args.test_aug_num = args.aug_num

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

    if args.problem in ("cvrp", "sdvrp"):
        args.CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}

    _STATE_K_TABLE = [35, 50, 65]
    args.state_k   = _STATE_K_TABLE[: args.num_state_encoder]

    tot_time_ckpt = 0.0
    start_epoch   = 0

    if args.finetune:
        print(f"\n[finetune] Loading checkpoint: {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=device)

        sd = ckpt.get("model_baseline") or ckpt.get("model_train")
        if sd is None:
            raise KeyError(
                "Checkpoint has neither 'model_baseline' nor 'model_train'. "
                f"Keys found: {list(ckpt.keys())}"
            )

        arch = _infer_arch_from_state_dict(sd, fallback=args)
        print("[finetune] Architecture auto-detected:")
        for k, v in arch.items():
            cli = getattr(args, k, "—")
            note = "  ← overrides CLI" if v != cli else ""
            print(f"           {k:35s} = {v}{note}")

        for k, v in arch.items():
            setattr(args, k, v)
        args.state_k = _STATE_K_TABLE[: args.num_state_encoder]

        model_train    = _build_model(arch, args.problem).to(device)
        model_baseline = _build_model(arch, args.problem).to(device)
        model_train.load_state_dict(sd)
        model_baseline.load_state_dict(
            ckpt.get("model_baseline") or ckpt.get("model_train")
        )

        optimizer = torch.optim.AdamW(model_train.parameters(), lr=args.model_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=args.gamma
        )

        if not args.reset_optimizer and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("[finetune] Optimizer state restored.")
            except Exception as e:
                print(f"[finetune] Could not restore optimizer state: {e}")
                print("[finetune] Starting with fresh optimizer.")
        else:
            print("[finetune] Starting with fresh optimizer.")

        if not args.reset_scheduler and "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                print("[finetune] Scheduler state restored.")
            except Exception:
                pass

        tot_time_ckpt = ckpt.get("tot_time", 0.0)
        print(
            f"[finetune] Resumed from epoch {ckpt.get('epoch', '?')}, "
            f"total wall-clock time so far: {tot_time_ckpt/3600:.1f}h"
        )
        
        ckpt_lambda = ckpt.get("rtdl_lambda", 0.0)
        if args.rtdl_lambda > 0 and ckpt_lambda == 0:
            print("[finetune] NOTE: checkpoint was trained WITHOUT RTDL; "
                  "now fine-tuning WITH RTDL. Consider --reset_optimizer.")
        elif args.rtdl_lambda == 0 and ckpt_lambda > 0:
            print("[finetune] NOTE: checkpoint was trained WITH RTDL; "
                  "now fine-tuning WITHOUT RTDL.")

    else:
        arch = dict(
            dim_input_nodes          = args.dim_input_nodes,
            dim_emb                  = args.dim_emb,
            dim_ff                   = args.dim_ff,
            num_state_encoder        = args.num_state_encoder,
            nb_layers_state_encoder  = args.nb_layers_state_encoder,
            nb_layers_action_encoder = args.nb_layers_action_encoder,
            nb_layers_decoder        = args.nb_layers_decoder,
            nb_heads                 = args.nb_heads,
            batchnorm                = args.batchnorm,
            if_agg_whole_graph       = args.if_agg_whole_graph,
        )
        model_train    = _build_model(arch, args.problem).to(device)
        model_baseline = _build_model(arch, args.problem).to(device)
        optimizer = torch.optim.AdamW(model_train.parameters(), lr=args.model_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=args.gamma
        )

    model_baseline.eval()
    n_params = sum(p.numel() for p in model_train.parameters())
    print(f"[model] parameters : {n_params:,}")
    print(args)

    log_dir = os.path.join(args.data_path, "ckpt", args.problem, "train", "logs")
    os.makedirs(log_dir, exist_ok=True)

    time_stamp = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    rtdl_tag   = "_rtdl" if args.rtdl_lambda > 0 else ""
    ft_tag     = "_ft"   if args.finetune else ""
    log_name   = (
        f"{time_stamp}"
        f"-n{args.nb_nodes}"
        f"-gpu{args.gpu_id}"
        f"{rtdl_tag}{ft_tag}.txt"
    )
    log_path = os.path.join(log_dir, log_name)
    file     = open(log_path, "w", 1)
    file.write(time_stamp + "\n\n")
    for arg in vars(args):
        file.write(f"{arg}={getattr(args, arg)}\n")
    file.write("\n\n")
    print(f"[log] {log_path}")

    train_model_with_knn(
        args, model_train, model_baseline,
        optimizer, scheduler,
        device, file, time_stamp,
    )

    if args.problem == "tsp":
        sizes         = [100, 1000, 5000, 10000]
        bszs          = [64,  32,   16,   8]
        num_instance  = [500, 50,   5,    5]
        distributions = ["uniform","clustered1","clustered2","explosion","implosion"]
        data_path     = os.path.join(args.data_path, "data/")
        run_tsp_test_knn(
            args.action_k, args.state_k, args.aug,
            model_baseline, args.if_use_local_mask,
            sizes, bszs, data_path, device, file,
            distributions, num_instance=num_instance,
        )
        run_tsplib_test_knn(model_baseline, args.action_k, args.state_k)

    elif args.problem == "cvrp":
        sizes         = [50, 500, 5000]
        bszs          = [64, 32,  16]
        num_instance  = [500, 50, 5]
        distributions = ["uniform","clustered1","clustered2","explosion","implosion"]
        data_path     = os.path.join(args.data_path, "data/")
        run_vrp_test_knn(
            args.action_k, args.state_k, args.aug,
            model_baseline, args.if_use_local_mask,
            sizes, bszs, data_path, device, file,
            distributions, num_instance,
        )
        run_cvrplib_test_knn(model_baseline, args.action_k, args.state_k)

    file.close()


if __name__ == "__main__":
    main()