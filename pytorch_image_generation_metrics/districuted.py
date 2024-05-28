import torch
import torch.distributed


def init(init_method, world_size, rank):
    torch.distributed.init_process_group('nccl', init_method, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()


def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def device():
    return torch.device(f'cuda:{rank()}')


def gather_shape(x: torch.Tensor, dim: int = 0):
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
    if world_size() > 1:
        shapes = gather_shape(x, cat_dim)
        xs = [torch.zeros(shape, device=x.device) for shape in shapes]
        torch.distributed.all_gather(xs, x)
        return torch.cat(xs, dim=cat_dim)
    else:
        return x


def sync_weight(model: torch.nn.Module):
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=0)
    for buffer in model.buffers():
        torch.distributed.broadcast(buffer.data, src=0)
    return model


def sync_grad(model: torch.nn.Module):
    for param in model.parameters():
        if param.requires_grad:
            torch.distributed.all_reduce(param.grad.data)
            param.grad.data /= world_size()
    return model


def print0(*args, **kwargs):
    if rank() == 0:
        print(*args, **kwargs)
