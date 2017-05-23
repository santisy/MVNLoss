from __future__ import print_function
import torch
from mvn import MVNLoss
from torch.autograd import Variable

# every row of params is a single sample, and every column of input is a single sample

def mvn_op(params, input):
    num_batch, num_params = params.size()
    dim = input.numel()/num_batch
    if params.is_cuda:
        output = Variable(torch.DoubleTensor(num_batch, 1).cuda())
    else:
        output = Variable(torch.DoubleTensor(num_batch, 1))

    for i in xrange(num_batch):
        mvn = MVNLoss()
        output[i, 0] = mvn(params[i, :].view(1, num_params), input[:, i].view(dim, 1))

    return output



