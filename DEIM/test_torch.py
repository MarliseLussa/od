import os
import torch

torch.distributed.init_process_group(backend='nccl', rank=0, world_size=4, init_method="tcp://127.0.0.1:29502")
torch.distributed.init_process_group(backend='nccl', rank=1, world_size=4, init_method="tcp://127.0.0.1:29502")
torch.distributed.init_process_group(backend='nccl', rank=2, world_size=4, init_method="tcp://127.0.0.1:29502")
torch.distributed.init_process_group(backend='nccl', rank=3, world_size=4, init_method="tcp://127.0.0.1:29502")

# local_rank  = os.getenv('RANK', -1)
# # print(local_rank)
# print(os.getenv('LOCAL_RANK', -1))
# print(os.getenv('WORLD_SIZE', 1))
        # RANK = int(os.getenv('RANK', -1))
        # LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
        # WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))