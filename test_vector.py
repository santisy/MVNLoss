import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from mvn import MVNLoss

# Test two dimension MVN
input = (Variable(torch.Tensor([0, 0, 8, 2, 8]).double().view(1, 5).cuda(), requires_grad=True),
         Variable(torch.Tensor([1, 1]).double().view(2, 1).cuda(), requires_grad=True))

test = gradcheck(MVNLoss(), input, eps=1e-6, atol=1e-4)

print("gradient check of two-dimension test: {}".format(test))

# Test three dimension MVN
input = (Variable(torch.Tensor([0, 0, 0, 5, 2, 1, 3, 1, 2]).double().view(1, 9), requires_grad=True),
         Variable(torch.Tensor([1, 1, 1]).double().view(3, 1), requires_grad=True))


test = gradcheck(MVNLoss(), input, eps=1e-6, atol=1e-4)

print("gradient check of Three-dimension test: {}".format(test))

# explicitly show the actual calculated forward result and backward gradient
input = (torch.Tensor([0, 0, 8, 2, 8]).double().view(1, 5),
         torch.Tensor([1, 1]).double().view(2, 1))

if torch.cuda.is_available():
    input = list(input)
    for i, x in enumerate(input):
        input[i] = x.cuda()

    input = tuple(input)

mvnLoss = MVNLoss()

print(mvnLoss.forward(input[0], input[1]))
print(mvnLoss.backward(torch.Tensor([[1.0,]])))

