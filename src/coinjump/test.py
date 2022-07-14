import torch

a = torch.tensor([[[-1,1,0,6],[-5,2,0,0],[0,3,0,0],[0,4,0,11]]])
b = torch.sum(a, 1)
print(a)