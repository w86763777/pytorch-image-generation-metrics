"""Utilities for distributed processing in PyTorch."""

import torch
import torch.distributed


def init(init_method, world_size, rank):
    """Initialize the distributed process group for multi-GPU processing.

    Args:
        init_method (str): URL specifying how to initialize the process group.
        world_size (int): Number of processes participating in the job.
        rank (int): Rank of the current process.

    Initializes the NCCL backend for distributed GPU communication and sets the current CUDA device to the given rank.
    """
    torch.distributed.init_process_group('nccl', init_method, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()


def rank():
    """Return the rank of the current process in the distributed process group.

    Returns:
        int: Rank of the current process. Returns 0 if the process group is not initialized.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def world_size():
    """Return the number of processes in the distributed process group.

    Returns:
        int: Number of processes. Returns 1 if the process group is not initialized.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def barrier():
    """Synchronize all processes in the distributed process group.

    Blocks until all processes have reached this function call.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def device():
    """Return the current CUDA device for the process based on its rank.

    Returns:
        torch.device: CUDA device object for the current process.
    """
    return torch.device(f'cuda:{rank()}')


def gather_shape(x: torch.Tensor, dim: int = 0):
    """Gather the shapes of tensors along a specific dimension from all processes in the distributed process group.

    Args:
        x (torch.Tensor): The tensor whose shape to gather.
        dim (int): The dimension along which to gather the shapes. Default is 0.

    Returns:
        list of torch.Size: A list of shapes from all processes.
    """
    if world_size() > 1:
        sizes_at_dim = [torch.tensor(0).to(x.device) for _ in range(world_size())]
        torch.distributed.all_gather(sizes_at_dim, torch.tensor(x.shape[dim], device=x.device))
        shapes = []
        for size in sizes_at_dim:
            shape = list(x.shape)
            shape[dim] = size.item()
            shapes.append(torch.Size(shape))
        return shapes
    else:
        return [x.shape]


def gather(x: torch.Tensor, cat_dim: int = 0):
    """Gather tensors from all processes and concatenates them along a specified dimension.

    Args:
        x (torch.Tensor): The tensor to gather.
        cat_dim (int): The dimension along which to concatenate the tensors. Default is 0.

    Returns:
        torch.Tensor: The concatenated tensor from all processes.
    """
    if world_size() > 1:
        shapes = gather_shape(x, cat_dim)
        xs = [torch.zeros(shape, device=x.device) for shape in shapes]
        torch.distributed.all_gather(xs, x)
        return torch.cat(xs, dim=cat_dim)
    else:
        return x


def print0(*args, **kwargs):
    """Print messages only from the process with rank 0."""
    if rank() == 0:
        print(*args, **kwargs)
