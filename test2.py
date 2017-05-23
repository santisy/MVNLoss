import torch
from torch.autograd import Variable
from mvn_op import mvn_op

params = Variable(torch.DoubleTensor([[0, 2], [0, 3]]), requires_grad=True).cuda()
input  = Variable(torch.DoubleTensor([[1, 1],]), requires_grad=True).cuda()

output = mvn_op(params, input)

result = torch.sum(output)

result.backward()

print(result)
