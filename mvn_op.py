from __future__ import print_function
import torch
from mvn import MVNLoss
from torch.autograd import Variable

# every row of params is a single sample, and every column of input is a single sample

def mvn_op(cov_chunk, miu_chunk, input):
    num_batch = len(cov_chunk)
    dim = cov_chunk[0].size(0)
    if cov_chunk[0].is_cuda:
        output = Variable(torch.DoubleTensor(num_batch, 1).cuda())
    else:
        output = Variable(torch.DoubleTensor(num_batch, 1))

    for i in xrange(num_batch):
        mvn = MVNLoss()
        cov_temp = cov_chunk[i, :, :]
        cov_temp = cov_temp + cov_temp.t() -torch.diag(torch.diag(cov_temp))
        cov_temp = torch.mm(cov_temp, cov_temp) + Variable(torch.eye(dim, dim).cuda())
        output[i, 0] = mvn(miu_chunk[i, :].unsqueeze(0), cov_temp, input[i, :].unsqueeze(0))

    return output



