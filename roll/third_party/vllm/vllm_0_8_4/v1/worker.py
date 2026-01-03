import gc
from typing import Optional

import torch
from vllm.v1.worker.gpu_worker import Worker
from vllm.device_allocator.cumem import CuMemAllocator

from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import RecvBucketManager

logger = get_logger()


class Worker084(WorkerHelper, Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def broadcast_bucket(self, src_pp_rank, meta_infos, bucket_size):
        RecvBucketManager.dict_to_meta(meta_infos)
        super().broadcast_bucket(src_pp_rank, meta_infos, bucket_size)

    def update_parameter(self, parameter_name, weight, ranks_in_worker):
        # Convert serialized weight dict back to tensor
        if isinstance(weight, dict) and "shape" in weight and "data" in weight:
            # Unpack shape and flat data list
            shape = tuple(weight["shape"])
            data = weight["data"]
            # Reconstruct tensor from flat list
            weight = torch.tensor(data, dtype=torch.float32, device='cuda').reshape(shape).to(torch.bfloat16)
        elif not isinstance(weight, torch.Tensor):
            # Fallback for other formats
            weight = torch.tensor(weight, dtype=torch.bfloat16, device='cuda')
        super().update_parameter(parameter_name, weight, ranks_in_worker)

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        RecvBucketManager.dict_to_meta(meta_infos)
        buffer = torch.tensor(buffer, dtype=torch.int8, device='cuda')
        super().update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)
