# MVNLoss
Try to realize Multivariate Normal Distribution loss in PyTorch

Now everything looks fine except there exists two latent problems:

1. In order to leverage the indexing mechanism of `numpy` to construct the covariance matrix, I manually copy data from GPU to CPU, 
if there exists a simple way to do it directly on GPU, it may favor the speed of this operation.

2. It is assumed that the covariance matrix is positive definite when calculating the probability, 
however, from the angle of output from certain NN during training, it may not be the case. 
