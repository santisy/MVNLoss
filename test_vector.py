import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from mvn import MVNLoss

# Test two dimension MVN
input = (Variable(torch.Tensor([0, 0]).double().view(1, 2).cuda(), requires_grad=True),
         Variable(torch.Tensor([[8, 2],[2, 8]]).double().cuda(), requires_grad=True),
         Variable(torch.Tensor([1, 1]).double().view(1, 2).cuda(), requires_grad=True))

test = gradcheck(MVNLoss(), input, eps=1e-6, atol=1e-4)

print("gradient check of two-dimension test: {}".format(test))

# Test three dimension MVN
input = (Variable(torch.Tensor([0, 0, 0]).double().view(1, 3), requires_grad=True),
         Variable(torch.Tensor([[5, 2, 1],[2, 8, 1],[1, 1, 3]]).double(), requires_grad=True),
         Variable(torch.Tensor([1, 1, 1]).double().view(1, 3), requires_grad=True))


test = gradcheck(MVNLoss(), input, eps=1e-6, atol=1e-4)

print("gradient check of Three-dimension test: {}".format(test))

# explicitly show the actual calculated forward result and backward gradient
input = (torch.Tensor([0, 0]).view(1, 2),
         torch.Tensor([[8, 2],[2, 8]]),
         torch.Tensor([1, 1]).view(1, 2))

if torch.cuda.is_available():
    input = list(input)
    for i, x in enumerate(input):
        input[i] = x.cuda()

    input = tuple(input)

mvnLoss = MVNLoss()

print(mvnLoss.forward(input[0], input[1], input[2]))
print(mvnLoss.backward(torch.Tensor([[1.0,]])))

