import torch



merge_inx = (torch.rand(50, 1)-0.48).ceil()

print(merge_inx.sum())