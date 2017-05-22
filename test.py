import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from learn_basic import MVNLoss

# Test one dimension MVN
# input = (Variable(torch.Tensor([0, 4]).double().view(1, 2), requires_grad=True),
#          Variable(torch.Tensor([1,]).double().view(1, 1), requires_grad=True))
#
# test = gradcheck(MVNLoss(), input, eps=1e-6, atol=1e-4)

# print("gradient check of one-dimension test: {}".format(test))

# Test two dimension MVN
input = (Variable(torch.Tensor([0, 0, 8, 2, 8]).double().view(1, 5), requires_grad=True),
         Variable(torch.Tensor([1, 1]).double().view(2, 1), requires_grad=True))

test = gradcheck(MVNLoss(), input, eps=1e-6, atol=1e-4)

print("gradient check of two-dimension test: {}".format(test))

# explicitly show the actual calculated forward result and backward gradient
# input = (torch.Tensor([0, 0, 8, 2, 8]).double().view(1, 5),
#          torch.Tensor([1, 1]).double().view(2, 1))
#
# mvnLoss = MVNLoss()
#
# print(mvnLoss.forward(input[0], input[1]))
# print(mvnLoss.backward(torch.Tensor([[1.0,]])))


# input = (torch.Tensor([0, 3]).double().view(1, 2),
#          torch.Tensor([1,]).double().view(1, 1))
#
# mvnLoss = MVNLoss()
#
# print(mvnLoss.forward(input[0], input[1]))
# print(mvnLoss.backward(torch.Tensor([[1.0,]])))
