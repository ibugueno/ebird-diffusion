from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, local_rank, world_size, device)


def finish(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def mean_across_processes(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, device=context.device, dtype=torch.float64)
    if context.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= context.world_size
    return float(tensor.item())
