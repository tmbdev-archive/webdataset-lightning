import torch

print("init")
torch.distributed.init_process_group("gloo")
print("done", torch.distributed.get_rank(), torch.distributed.get_world_size())
