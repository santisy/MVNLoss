from __future__ import print_function
import torch
from torch.autograd import Variable
from mvn_op import mvn_op

params = Variable(torch.DoubleTensor([[0, 0, 8, 2, 8], [0, 0, 8, 2, 8]]).cuda(), requires_grad=True)
input = Variable(torch.DoubleTensor([[1, 1], [1, 1]]).cuda(), requires_grad=True)

output = mvn_op(params, input)

result = torch.sum(output)

result.backward()

print(result)

print(params.grad)
print(input.grad)
