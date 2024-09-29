# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import functools
import io
import logging
import os
import random
import tempfile
import time
from typing import Any, Callable, List, Tuple

import torch
import torch.autograd as autograd
import torch.distributed as dist


# Default to GPU 0
_cuda_device_index: int = 0

# Setting _cuda_device_index to -1 internally implies that we should use CPU
_CPU_DEVICE_INDEX = -1
_PRIMARY_RANK = 0


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """

    if dist.get_backend() == "nccl":
        # Increase timeout from 1800 sec to 43200 sec (12 hr) to avoid some processes
        # being much slower than others causing a timeout (which can happen in relation
        # or LVIS class mAP evaluation).
        timeout = 43200
        return dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=timeout),
        )

    return dist.group.WORLD


def is_main_process():
    """Return true if the current process is the main one"""
    return get_rank() == 0


def all_gather_via_filesys(data, filesys_save_dir=None, gather_to_rank_0_only=False):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors), similar to
    `all_gather` above, but using filesystem instead of collective ops.

    If gather_to_rank_0_only is True, only rank 0 will load the gathered object list
    (and other ranks will have an empty list).
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    print("gathering via files")
    cpu_group = _get_global_gloo_group()

    # if unspecified, we will save to the current python file dir
    if filesys_save_dir is not None:
        save_dir = filesys_save_dir
    elif "EXP_DIR" in os.environ:
        save_dir = os.environ["EXP_DIR"]
    else:
        # try the same directory where the code is stored
        save_dir = filesys_save_dir or os.path.dirname(__file__)
    save_dir = os.path.join(save_dir, "all_gather_via_filesys")
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    # use a timestamp and salt to distinguish different all_gather
    timestamp = int(time.time()) if is_main_process() else 0
    salt = random.randint(0, 2**31 - 1) if is_main_process() else 0
    # broadcast the timestamp and salt across ranks
    # (all-reduce will do the broadcasting since only rank 0 is non-zero)
    timestamp_and_salt = torch.tensor([timestamp, salt], dtype=torch.long)
    dist.all_reduce(timestamp_and_salt, group=cpu_group)
    timestamp, salt = timestamp_and_salt.tolist()

    # save the data to a file on the disk
    rank_save = get_rank()
    save_data_filename = f"data_to_gather_{timestamp}_{salt}_{rank_save}.pkl"
    save_data_path = os.path.join(save_dir, save_data_filename)
    assert not os.path.exists(save_data_path), f"{save_data_path} already exists"
    torch.save(data, save_data_path)
    dist.barrier(group=cpu_group)

    # read the data from the files
    data_list = []
    if rank_save == 0 or not gather_to_rank_0_only:
        for rank_load in range(world_size):
            load_data_filename = f"data_to_gather_{timestamp}_{salt}_{rank_load}.pkl"
            load_data_path = os.path.join(save_dir, load_data_filename)
            assert os.path.exists(load_data_path), f"cannot read {save_data_path}"
            data_list.append(torch.load(load_data_path))
    dist.barrier(group=cpu_group)

    # delete the saved file
    os.remove(save_data_path)
    return data_list


def all_gather(data, force_cpu=False, force_filesys=False, filesys_save_dir=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    if os.getenv("MDETR_FILESYS_REDUCE_RANK_0_ONLY") == "1":
        return all_gather_via_filesys(
            data, filesys_save_dir, gather_to_rank_0_only=True
        )

    if os.getenv("MDETR_FILESYS_REDUCE") == "1" or force_filesys:
        return all_gather_via_filesys(data, filesys_save_dir)

    cpu_group = None
    if os.getenv("MDETR_CPU_REDUCE") == "1" or force_cpu:
        cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [
        torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)
    ]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device=device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy())
        obj = torch.load(buffer)
        data_list.append(obj)

    return data_list


def convert_to_distributed_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor: torch.Tensor, orig_device: str) -> torch.Tensor:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor


def is_distributed_training_run() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


def is_primary() -> bool:
    """
    Returns True if this is rank 0 of a distributed training job OR if it is
    a single trainer job. Otherwise False.
    """
    return get_rank() == _PRIMARY_RANK


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing mean reduction
    of tensor over all processes.
    """
    return all_reduce_op(
        tensor,
        torch.distributed.ReduceOp.SUM,
        lambda t: t / torch.distributed.get_world_size(),
    )


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing sum
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.SUM)


def all_reduce_min(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing min
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MIN)


def all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing min
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MAX)


def all_reduce_op(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    after_op_func: Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.all_reduce(tensor, op)
        if after_op_func is not None:
            tensor = after_op_func(tensor)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def gather_tensors_from_all(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Wrapper over torch.distributed.broadcast for broadcasting a tensor from the source
    to all processes in both distributed / non-distributed scenarios.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.broadcast(tensor, src)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def barrier() -> None:
    """
    Wrapper over torch.distributed.barrier, returns without waiting
    if the distributed process group is not initialized instead of throwing error.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    torch.distributed.barrier()


def get_world_size() -> int:
    """
    Simple wrapper for correctly getting worldsize in both distributed
    / non-distributed settings
    """
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )


def get_rank() -> int:
    """
    Simple wrapper for correctly getting rank in both distributed
    / non-distributed settings
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )


def get_primary_rank() -> int:
    return _PRIMARY_RANK


def set_cuda_device_index(idx: int) -> None:
    global _cuda_device_index
    _cuda_device_index = idx
    torch.cuda.set_device(_cuda_device_index)


def set_cpu_device() -> None:
    global _cuda_device_index
    _cuda_device_index = _CPU_DEVICE_INDEX


def get_cuda_device_index() -> int:
    return _cuda_device_index


def init_distributed_data_parallel_model(
    model: torch.nn.Module,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = True,
    bucket_cap_mb: int = 25,
) -> torch.nn.parallel.DistributedDataParallel:
    global _cuda_device_index

    if _cuda_device_index == _CPU_DEVICE_INDEX:
        # CPU-only model, don't specify device
        return torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            bucket_cap_mb=bucket_cap_mb,
        )
    else:
        # GPU model
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[_cuda_device_index],
            output_device=_cuda_device_index,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            bucket_cap_mb=bucket_cap_mb,
        )


def broadcast_object(obj: Any, src: int = _PRIMARY_RANK, use_disk: bool = True) -> Any:
    """Broadcast an object from a source to all workers.

    Args:
        obj: Object to broadcast, must be serializable
        src: Source rank for broadcast (default is primary)
        use_disk: If enabled, removes redundant CPU memory copies by writing to
            disk
    """
    # Either broadcast from primary to the fleet (default),
    # or use the src setting as the original rank
    if get_rank() == src:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data_view = buffer.getbuffer()
        length_tensor = torch.LongTensor([len(data_view)])
        length_tensor = broadcast(length_tensor, src=src)
        data_tensor = torch.ByteTensor(data_view)
        data_tensor = broadcast(data_tensor, src=src)
    else:
        # Fetch from the source
        length_tensor = torch.LongTensor([0])
        length_tensor = broadcast(length_tensor, src=src)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8)
        data_tensor = broadcast(data_tensor, src=src)
        if use_disk:
            with tempfile.TemporaryFile("r+b") as f:
                f.write(data_tensor.numpy())
                # remove reference to the data tensor and hope that Python garbage
                # collects it
                del data_tensor
                f.seek(0)
                obj = torch.load(f)
        else:
            buffer = io.BytesIO(data_tensor.numpy())
            obj = torch.load(buffer)
    return obj


def all_gather_tensor(tensor: torch.Tensor, world_size=None):
    if world_size is None:
        world_size = get_world_size()
    # make contiguous because NCCL won't gather the tensor otherwise
    assert tensor.is_contiguous(), f"{tensor.shape} is not contiguous!"
    tensor, orig_device = convert_to_distributed_tensor(tensor)
    tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_all, tensor, async_op=False)  # performance opt
    tensor_all = [
        convert_to_normal_tensor(tensor, orig_device) for tensor in tensor_all
    ]
    return tensor_all


def all_gather_batch(tensors: List[torch.Tensor]):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = all_gather_tensor(tensor, world_size)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def create_new_process_group(group_size):
    """
    Creates process groups of a gives `group_size` and returns
    process group that current GPU participates in.

    `group_size` must divide the total number of GPUs (world_size).

    Modified from
    https://github.com/NVIDIA/apex/blob/4e1ae43f7f7ac69113ef426dd15f37123f0a2ed3/apex/parallel/__init__.py#L60

    Args:
        group_size (int): number of GPU's to collaborate for sync bn
    """

    assert group_size > 0

    world_size = torch.distributed.get_world_size()
    if world_size <= 8:
        if group_size > world_size:
            logging.warning(
                f"Requested group size [{group_size}] > world size [{world_size}]. "
                "Assuming local debug run and capping it to world size."
            )
            group_size = world_size
    assert world_size >= group_size
    assert world_size % group_size == 0

    group = None
    for group_num in range(world_size // group_size):
        group_ids = range(group_num * group_size, (group_num + 1) * group_size)
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if torch.distributed.get_rank() // group_size == group_num:
            group = cur_group
            # can not drop out and return here, every process must go through creation of all subgroups

    assert group is not None
    return group


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
