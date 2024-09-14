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

c = nn.Conv2d(1, 32, 5)
# with open("tensor.bin", "rb") as f:
#     c.weight = nn.Parameter(torch.frombuffer(f.read(32 * 1 * 5 * 5 * 4), dtype=torch.float32).reshape(32, 1, 5, 5).clone())
#     c.bias = nn.Parameter(torch.frombuffer(f.read(32 * 4), dtype=torch.float32).clone())

# print(c(torch.arange(0, 28*28, dtype=torch.float32).reshape(1, 1, 28, 28)).flatten()[:10])

# c = torch.arange(9, dtype=torch.float32)

w = c.weight.flatten().tolist()
b = c.bias.flatten().tolist()
import struct
ba = b"".join([struct.pack("f", v) for v in w] + [struct.pack("f", v) for v in b])

with open("tensor.bin", "wb") as f:
    f.write(ba)
