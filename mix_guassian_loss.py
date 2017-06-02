import torch
from torch.autograd import Variable
from mvn_op import mvn_op

def mg_loss(mvn_param, lat_var, bbox_input, dim=4):
    lat_num = lat_var.size(1)
    param_stride = int(dim*(dim+1)/2+dim)
    if torch.cuda.is_available():
        output = Variable(torch.zeros(lat_var.size()).cuda())
    else:
        output = Variable(torch.zeros(lat_var.size()))
    for i in range(lat_num):
        # now in order to make the covariance matrix posotive definite do C*C'
        mvn_param_chunk = mvn_param[:, i*param_stride:(i+1)*param_stride]
        miu_chunk = mvn_param_chunk[:, :dim]
        cov_flat_chunk = mvn_param_chunk[:, dim:param_stride]

        if torch.cuda.is_available():
            cov_chunk = Variable(torch.zeros((mvn_param.size(0), dim, dim)).cuda())
        else:
            cov_chunk = Variable(torch.zeros((mvn_param.size(0), dim, dim)))

        temp_count = 0
        for j in range(dim):
            cov_chunk[:, j:j+1, j:] = cov_flat_chunk[:, temp_count:temp_count+dim-j].unsqueeze(1)
            temp_count += dim - j

        output[:, i] = mvn_op(cov_chunk, miu_chunk, bbox_input)

    return -torch.sum(torch.sum(output.exp()*lat_var, 1).log())/lat_var.size(0)



