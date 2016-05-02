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

    def kernel(self, x, xstar, sess):
        x = sess.run(x)
        xstar = sess.run(xstar)
        s = np.power(self.eta, 2.0) *\
            np.exp(-1.0/(np.power(self.l,   2.0) * 2.0) * \
                         (np.sum(np.power(x - xstar , 2.0))))
        
        return s

    def compute_K(self, xs):
        sess = tf.Session()
        mat = np.zeros(shape=(self.N, self.N))
        for i in xrange(self.N):
            xi = xs[i, 1:]
            for j in xrange(self.N):
                if j == i:
                    mat[i, i] = self.kernel(xi, xi, sess)
                else:
                    sij = self.kernel(xi, xs[j, 1:], sess)
                    mat[i, j] = sij
        sess.close()
        return tf.constant(mat, dtype=tf.float32)
    
    def logit(self, x):
        
        return tf.truediv(1.0, (1.0 + tf.exp(-x))) 
                            
    def probit(self, x):
        
        return 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))    
    
    def sigmoid(self, x):
        "Numerically-stable sigmoid function."
        if x >= 0.0:
            z = tf.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = tf.exp(x)
            return z / (1.0 + z)

    def log_prob(self, xs, zs):
        K = self.compute_K(xs)
        log_prior = multivariate_normal.logpdf(zs[:, :], cov=K)
        log_lik = tf.concat(0, [tf.reduce_sum(bernoulli.logpmf(xs[:,0], 
                                self.sigmoid(tf.mul(xs[:,0], z)))) 
                                for z in tf.unpack(zs)])

        return log_prior + log_lik

    def predict():
        pass

    
def main():
    ed.set_seed(42)
    df = np.loadtxt('./gpc_data/crabs_train.txt', dtype='float32', delimiter=',')
    eta = 1.0
    l = 1.0
    N = len(df)
    data = ed.Data(tf.constant(df, dtype=tf.float32))
    model = GaussProcess(eta, l, N)
    variational = ed.MFGaussian(N)
    infer = ed.MFVI(model, variational, data)
    infer.run(n_iter=10000)

if __name__ == "__main__":
    main()
