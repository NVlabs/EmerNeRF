import builtins
import datetime
import json
import logging
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # convert numpy array to list
        return super(NumpyEncoder, self).default(obj)


def get_robust_pca(features: torch.Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    rins = colors[fg_mask][s[:, 0] < m, 0]
    gins = colors[fg_mask][s[:, 1] < m, 1]
    bins = colors[fg_mask][s[:, 2] < m, 2]

    rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
    rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def resume_from_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    proposal_networks: Optional[
        List[
            torch.nn.Module,
        ]
    ] = None,
    proposal_estimator: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """
    Load model from checkpoint.
    """
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    msg = model.load_state_dict(checkpoint["model"])
    logger.info(f"radiance_field: {msg}")
    if proposal_networks is not None:
        for i, p in enumerate(proposal_networks):
            msg = p.load_state_dict(checkpoint["proposal_networks"][i])
            logger.info(f"proposal_networks[{i}]: {msg}")
    if proposal_estimator is not None:
        if proposal_estimator.optimizer is not None:
            if "estimator.optimizer" in checkpoint:
                msg = proposal_estimator.optimizer.load_state_dict(
                    checkpoint["estimator.optimizer"]
                )
                logger.info(f"proposal_estimator.optimizer: {msg}")
        if proposal_estimator.scheduler is not None:
            if "estimator.scheduler" in checkpoint:
                msg = proposal_estimator.scheduler.load_state_dict(
                    checkpoint["estimator.scheduler"]
                )
                logger.info(f"proposal_estimator.scheduler: {msg}")
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "step" in checkpoint:
        logger.info(f"Will resume from step {checkpoint['step']}")
        return checkpoint["step"]
    else:
        logger.info("Will start from step 0")
        return 0


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
