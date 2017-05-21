import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from learn_basic import MVNLoss

input = (Variable(torch.Tensor([0, 0, 3, 2, 3]).double().view(1, 5), requires_grad=True),
         Variable(torch.Tensor([1, 1]).double().view(2, 1), requires_grad=True))

test = gradcheck(MVNLoss(), input, eps=1e-6, atol=1e-4)

print(test)

# input = (torch.Tensor([0, 0, 3, 2, 3]).double().view(1, 5),
#          torch.Tensor([1, 1]).double().view(2, 1))
#
# mvnLoss = MVNLoss()
#
# print(mvnLoss.forward(input[0], input[1]))
# print(mvnLoss.backward(torch.Tensor([[1.0,]])))
