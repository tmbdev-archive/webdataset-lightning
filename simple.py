import sys
import torch

print("init")
torch.distributed.init_process_group("gloo")
print("done", torch.ditsributed.get_rank(), torch.distributed.get_world_size())
