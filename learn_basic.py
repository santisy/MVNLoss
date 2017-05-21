from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Function


"""
TODO: 1. For now, not Support Batch
TODO: 2, the leverage of numpy shared memory here will not work in cuda tensor
"""

class MVNLoss(Function):

    def __init__(self):
        super(MVNLoss, self).__init__()

        self.dim = None
        self.input_minus_miu = None
        self.inv_sing_mat = None

    def forward(self, params, input):
        """
        
        :param ctx: 
        :param params: vector miu and unique elements of covariance matrix sigma
                       the first dim number of params is miu and then is sigma
        :param input: ground truth samples from the training set  
        :param dim: the dimension of the distribution
        :return: 
        """
        self.save_for_backward(params, input)

        dim = np.prod(input.size())
        param_dim = np.prod(params.size())

        assert (param_dim == (dim*dim/2.0+(3.0/2)*dim)), "dimension does not match"

        sigma = np.zeros((dim, dim))
        sigma_t = torch.from_numpy(sigma)
        iu = np.triu_indices(dim)
        sigma[iu] = params[0, dim:].numpy()
        sigma_t = sigma_t.t() + sigma_t
        sigma_t = sigma_t - torch.diag(torch.diag(sigma_t))/2.0

        # the miu
        miu = params[0, :dim].view(-1, dim)

        input_raw = input.view(-1, dim)
        input_minus_miu = input_raw - miu # 1xdim vector
        u_mat, sing_values, v_mat = torch.svd(sigma_t)
        inv_sing_mat = torch.mm(torch.mm(u_mat, torch.diag(1./sing_values)), v_mat.t())

        self.dim = dim
        self.input_minus_miu = input_minus_miu
        self.inv_sing_mat = inv_sing_mat

        output = - dim/2.0 * np.log(2*np.pi) + (-1.0/2) * np.log(torch.prod(sing_values)) + \
                 (-1.0/2)*torch.mm(torch.mm(input_minus_miu, inv_sing_mat), input_minus_miu.t())

        # the output is actually the log pdf
        return output


    def backward(self, grad_output):
        """
        Here exist two assumptions:
        1. the covariance matrix we construct from the output of RNN is positive definite
        2. the parameters we get from RNN through this operation is 1xdim, while the input is dimx1
        """
        # params, input = self.saved_variables
        grad_output = grad_output[0, 0]

        inv_sing_mat = self.inv_sing_mat
        input_minus_miu = self.input_minus_miu
        dim = self.dim

        grad_params = torch.zeros((1, int(dim*dim/2.0+(3.0/2)*dim)))

        # gradients of covariance matrix
        mid_result1 = torch.mm(input_minus_miu.t(), input_minus_miu)

        grad_sigma1 = -torch.mm(torch.mm(inv_sing_mat, mid_result1), inv_sing_mat)
        grad_sigma2 = inv_sing_mat
        grad_sigma = -(1.0/2)*(grad_sigma1 + grad_sigma2)*grad_output

        grad_sigma_np = grad_sigma.numpy()

        iu = np.triu_indices(dim)
        grad_params[0, dim:] = torch.Tensor(grad_sigma_np[iu])

        # gradients of miu
        grad_miu = -torch.mm(inv_sing_mat, -input_minus_miu.t())*grad_output

        grad_params[0, :dim] = grad_miu

        # gradients of input
        grad_input = -torch.mm(inv_sing_mat, input_minus_miu.t())*grad_output

        return grad_params, grad_input
