import torch.nn.functional as F
import torch

arr = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).unsqueeze(0).unsqueeze(0)

kernels = torch.tensor([[1, 2],
                    [-1, 0]]).unsqueeze(0).unsqueeze(0)

kernels_rot180 = torch.tensor([[0, -1],
                    [2, 1]]).unsqueeze(0).unsqueeze(0)

print(F.conv2d(arr, kernels))
print(F.conv2d(arr, kernels_rot180))