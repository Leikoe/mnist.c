import torch.nn.functional as F
import torch.nn as nn
import torch

# arr = torch.tensor([[1, 6, 2],
#                     [5, 3, 1],
#                     [7, 0, 4]]).unsqueeze(0).unsqueeze(0)

# kernels = torch.tensor([[1, 2],
#                     [-1, 0]]).unsqueeze(0).unsqueeze(0)

# kernels_rot180 = torch.tensor([[0, -1],
#                     [2, 1]]).unsqueeze(0).unsqueeze(0)

# print(F.conv2d(arr, kernels))
# print(F.conv2d(arr, kernels_rot180))

# c = nn.Conv2d(1, 32, 5)
c = torch.arange(9).float()

bs = c.tolist()
import struct
ba = b"".join([struct.pack("f", v) for v in bs])
print(bs)
print(ba)

with open("tensor.bin", "wb") as f:
    f.write(ba)