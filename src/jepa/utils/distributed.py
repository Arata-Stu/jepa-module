from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    backend: str
    rank: int
    world_size: int
    local_rank: int


def _default_local_rank(rank: int) -> int:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return rank % torch.cuda.device_count()
    return 0


def _select_backend(name: str) -> str:
    if name != "auto":
        return name
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_distributed(
    enabled: bool,
    backend: str = "auto",
    port: int = 37129,
    timeout_sec: int = 1800,
) -> DistributedState:
    if not enabled:
        return DistributedState(
            enabled=False,
            backend="none",
            rank=0,
            world_size=1,
            local_rank=0,
        )

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(_default_local_rank(rank))))
        return DistributedState(
            enabled=(world_size > 1),
            backend=str(dist.get_backend()),
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(_default_local_rank(rank))))
    if world_size <= 1:
        return DistributedState(
            enabled=False,
            backend="none",
            rank=0,
            world_size=1,
            local_rank=0,
        )

    master_addr = os.environ.get("MASTER_ADDR", "")
    if master_addr in {"", "localhost"}:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    master_port = os.environ.get("MASTER_PORT", "")
    if master_port in {"", "0"}:
        os.environ["MASTER_PORT"] = str(port)

    resolved_backend = _select_backend(backend)
    dist.init_process_group(
        backend=resolved_backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=int(timeout_sec)),
    )
    return DistributedState(
        enabled=True,
        backend=resolved_backend,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process(state: DistributedState | None = None) -> bool:
    if state is not None:
        return state.rank == 0
    if not is_dist_initialized():
        return True
    return dist.get_rank() == 0


def barrier() -> None:
    if is_dist_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def reduce_mean_tensor(x: torch.Tensor) -> torch.Tensor:
    if not is_dist_initialized() or dist.get_world_size() <= 1:
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= dist.get_world_size()
    return y


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor(value, dtype=torch.float32, device=device)
    return float(reduce_mean_tensor(t).item())


def unwrap_module(module: Any) -> Any:
    return module.module if hasattr(module, "module") else module


def cleanup_distributed() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()
