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

    def update_parameter(self, parameter_name, weight, ranks_in_worker, dtype_str=None, shape=None):
        # Reconstruct tensor from list if serialized
        if dtype_str is not None and shape is not None:
            # Map dtype string to torch dtype
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            dtype = dtype_map.get(dtype_str, torch.float32)
            weight = torch.tensor(weight, dtype=dtype, device='cuda').reshape(shape)
        super().update_parameter(parameter_name, weight, ranks_in_worker)

    def update_parameter_in_bucket(self, meta_infos, buffer, ranks_in_worker):
        RecvBucketManager.dict_to_meta(meta_infos)
        buffer = torch.tensor(buffer, dtype=torch.int8, device='cuda')
        super().update_parameter_in_bucket(meta_infos, buffer, ranks_in_worker)
