
import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Variational, Normal
from edward.stats import norm, poisson


class Latent_Space_Model():
    """
    p(x, z) = [ prod_{i=1}^N prod_{j=1}^N Poi(Y_{ij}; \exp(s_iTt_j) ) ]
              [ prod_{i=1}^N N(s_i; 0, var) N(t_i; 0, var) ]
              
    where z = {s,t}.
    """
    
    def __init__(self,N,K,var=0.01, 
                 like ='Poisson', 
                 prior='Lognormal', 
                 dist = 'euclidean',
                 interaction ='additive'):
        self.num_vars = N * K
        self.N = N
        self.K = K
        self.prior_variance = var
        self.like = like
        self.prior = prior
        self.interaction = interaction
        self.dist = dist


    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        if self.prior == 'Lognormal':
            zs = tf.exp(zs)
        elif self.prior != 'Gaussian':
            raise NotImplementedError("prior not available.")
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs)

        z = tf.reshape(zs,[self.N,self.K])

        if self.dist == 'euclidean':
            xp = tf.matmul(tf.ones([1,self.N]),tf.reduce_sum(z*z,1,keep_dims=True))
            xp = xp + tf.transpose(xp) - 2*tf.matmul(z,z,transpose_b = True)
        elif self.dist == 'cosine':
            xp = tf.matmul(z,z,transpose_b = True)
        if self.interaction == 'multiplicative':
            xp = tf.exp(xp)
        elif self.interaction != 'additive':
            raise NotImplementedError("interaction type unknown.")
        if self.like == 'Gaussian':
            log_lik = tf.reduce_sum(norm.logpdf(xs,xp))
        elif self.like == 'Poisson':
            if not (self.interaction == "additive" or self.prior == "Lognormal"):
                raise NotImplementedError("Rate of Poisson has to be nonnegatve.")
            log_lik = tf.reduce_sum(poisson.logpmf(xs,xp))
        else:
            raise NotImplementedError("likelihood not available.")
        return log_lik + log_prior

def load_celegans_brain():
    data = np.load('data/celegans_brain.npy')
    N = data.shape[0]
    return ed.Data(data), N

ed.set_seed(42)
data, N = load_celegans_brain()
K = 3 
model = Latent_Space_Model(N,K,
                           like='Gaussian',
                           prior='Gaussian',
                           interaction='additive')

inference = ed.MAP(model, data)

#variational = Variational()
#variational.add(Normal(model.num_vars))
#inference = ed.MFVI(model, variational,data)

var = inference.run(n_iter=5000, n_print=500)
