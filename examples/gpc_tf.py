#!/usr/bin/env python                                                                                                                 
"""                                                                                                                                   
Gaussian Process Classification                                                                                                       
---------------------------------                                                                                                     
Data: y(label), x(covariates)                                                                                                         
Latent: function f such that y = f(x)                                                                                                
Prior on f: N(0, K) where K is parameterized by \theta = (\eta, l)                                                                
Likelihood: p(y | f) = \prod_{i=1}^{N} \Phi(yi * fi) 
where fi = f(xi) and \Phi is the cdf of N(0,1) or sigmoid                       
Posterior: q(f | D, \theta) = N(m, A)                                                                                                

We approximate m and A via chiVI and compare the performance of the algo                                                              
against EP, VI and Laplace on benchmark datasets using log likelihood estimation                                                      
and predictive performance                                                                                             
The model is implemented in Tensorflow                                                                                                

Remarks: this code requires you to have the labels in the first column and the covariates 
in subsequent columns
"""


import tensorflow as tf
import edward as ed
import numpy as np
from edward.stats import multivariate_normal, bernoulli
import matplotlib.pyplot as plt

class GaussProcess:
    def __init__(self, eta, l, N):
        self.eta = eta
        self.l = l
        self.N = N
        self.num_vars = N

    def kernel(self, xs):
        mat = tf.zeros(shape=(self.N, self.N), dtype=tf.float32)
        for i in xrange(self.N):
            for j in xrange(i+1, self.N, 1):
                sij = tf.concat(0,[tf.pow(self.eta, 2.0) *\
                                   tf.exp(-1.0/(tf.pow(self.l, 2.0) * 2.0) *\
                                   (tf.reduce_sum(tf.pow(xs[i, 1:] - xs[j, 1:] , 2.0))))])
                sij = tf.reshape(sij, (1,))
                shape = [self.N, self.N]
                indices1 = [[i, j]]
                indices2 = [[j, i]]
                delta1 = tf.SparseTensor(indices1, sij, shape)
                delta2 = tf.SparseTensor(indices2, sij, shape)
                mat = mat + tf.sparse_tensor_to_dense(delta1)
                mat = mat + tf.sparse_tensor_to_dense(delta2)
        indices = [[i, i]]
        shape = [self.N, self.N]
        value = tf.reshape(tf.pow(self.eta, 2.0), (1,))
        delta = tf.SparseTensor(indices, value, shape)
        mat = mat + tf.sparse_tensor_to_dense(delta)

        return mat

    def logit(self, x, z):
        """x[:, 0] is y """
        return tf.truediv(1.0, (1.0 + tf.exp(-x * z))) 
                            
    def probit(self, x, z):
        
        return 0.5 * (1.0 + tf.erf(x * z / tf.sqrt(2.0)))    
                            
    def log_prob(self, xs, zs):
        ker = self.kernel(xs)
        log_prior = multivariate_normal.logpdf(zs[:, :], cov=ker)
        log_lik = tf.concat(0, [tf.reduce_sum(bernoulli.logpmf(xs[:,0], 
                                self.probit(xs[:,0], z))) 
                                for z in tf.unpack(zs)])

        return log_prior + log_lik

    def predict():
        pass

    def error():
        pass

    def boundary():
        pass
    
def main():
    ed.set_seed(42)
    #df = np.loadtxt('./gpc_data/banana_train.txt', dtype='float32', delimiter=',')[0:400,:]
    df = np.loadtxt('./gpc_data/crabs_train.txt', dtype='float32', delimiter=',')
    eta = 1.0
    l = 1.0
    N = len(df)
    data = ed.Data(tf.constant(df, dtype=tf.float32))
    model = GaussProcess(eta, l, N)
    variational = ed.MFGaussian(N)
    infer_kl = ed.MFVI(model, variational, data)
    infer_chi = ed.ChiVI(model, variational, data)
    infer_kl.run(n_iter=1000)
    infer_chi.run(n_iter=1000)

if __name__ == "__main__":
    main()
