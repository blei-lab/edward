#!/usr/bin/env python
"""Linear mixed effects model using lme4::InstEval instructor rating data.

Fits y ~ 1 + (1|s) + (1|d) + service + (1|dept)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import pandas as pd
import tensorflow as tf

from edward.models import Normal


data = pd.read_csv('data/insteval.csv')

ed.set_seed(42)

# DATA
ytrain = data['y'].values.astype(float)
# s - students - 1:2972
s = data['s'].values.astype(int) - 1
# d - instructors - codes that need to be remapped
data['dcodes'] = data['d'].astype('category').cat.codes
d = data['dcodes'].values.astype(int)
# dept also needs to be remapped
data['deptcodes'] = data['dept'].astype('category').cat.codes
dept = data['deptcodes'].values.astype(int)
service = data['service'].values

n_s = 2972
n_d = 1128
n_dept = 14
n_obs = data.shape[0]

# MODEL
lnvar_s = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
lnvar_d = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
lnvar_dept = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

sigma_s = tf.sqrt(tf.exp(lnvar_s))
sigma_d = tf.sqrt(tf.exp(lnvar_d))
sigma_dept = tf.sqrt(tf.exp(lnvar_dept))

mu = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
service = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

eta_s = Normal(mu=tf.zeros(n_s),
               sigma=sigma_s*tf.ones(n_s))
eta_d = Normal(mu=tf.zeros(n_d),
               sigma=sigma_d*tf.ones(n_d))
eta_dept = Normal(mu=tf.zeros(n_dept),
                  sigma=sigma_dept*tf.ones(n_dept))

yhat = tf.gather(eta_s, s) + \
       tf.gather(eta_d, d) + \
       tf.gather(eta_dept, dept) + \
       mu + service
y = Normal(mu=yhat,
           sigma=tf.ones(n_obs))


# INFERENCE
def make_normal(n):
    var = Normal(
        mu=tf.Variable(tf.random_normal([n])),
        sigma=tf.nn.softplus(tf.Variable(tf.random_normal([n]))))
    return var


q_eta_s = make_normal(n_s)
q_eta_d = make_normal(n_d)
q_eta_dept = make_normal(n_dept)

qlnvar_s = make_normal(1)
qlnvar_d = make_normal(1)
qlnvar_dept = make_normal(1)

qmu = make_normal(1)
qservice = make_normal(1)

params_dict = {
    mu: qmu,
    service: qservice,
    lnvar_s: qlnvar_s,
    lnvar_d: qlnvar_d,
    lnvar_dept: qlnvar_dept,
    eta_s: q_eta_s,
    eta_d: q_eta_d,
    eta_dept: q_eta_dept
}

data_dict = {y: ytrain}

print('Making inference')
inference = ed.KLqp(params_dict, data_dict)

inference.run(n_iter=1000)
