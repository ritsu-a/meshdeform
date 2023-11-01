import torch
import numpy as np
import ipdb


ipdb.set_trace()
x = torch.randn(2, 2, requires_grad = True)

x = np.array([1., 2., 3.]) #Only Tensors of floating point dtype can require gradients
x = torch.from_numpy(x)

x.requires_grad_(True)
