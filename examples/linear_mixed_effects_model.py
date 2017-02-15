#!/usr/bin/env python
"""Linear mixed effects model using lme4::InstEval instructor rating data.

Fits y ~ 1 + (1|s) + (1|d) + service + (1|dept)

Data described at https://cran.r-project.org/web/packages/lme4/lme4.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from edward.models import Normal

data = pd.read_csv('data/insteval.csv')
ed.set_seed(42)


# DATA
# s - students - 1:2972
# d - instructors - codes that need to be remapped
# dept also needs to be remapped
data['dcodes'] = data['d'].astype('category').cat.codes
data['deptcodes'] = data['dept'].astype('category').cat.codes
data['s'] = data['s'] - 1

train = data.sample(frac=0.8)
test = data.drop(train.index)

s_train = train['s'].values.astype(int)
d_train = train['dcodes'].values.astype(int)
dept_train = train['deptcodes'].values.astype(int)
y_train = train['y'].values.astype(float)
service_train = train['service'].values.astype(int)
n_obs_train = train.shape[0]
service_train.shape = [n_obs_train, 1]


s_test = test['s'].values.astype(int)
d_test = test['dcodes'].values.astype(int)
dept_test = test['deptcodes'].values.astype(int)
y_test = test['y'].values.astype(float)
service_test = test['service'].values.astype(int)
n_obs_test = test.shape[0]
service_test.shape = [n_obs_test, 1]

n_s = 2972
n_d = 1128
n_dept = 14
n_obs = train.shape[0]

# MODEL
service_X = tf.placeholder(tf.float32, [n_obs, 1])

lnvar_s = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
lnvar_d = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
lnvar_dept = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

sigma_s = tf.sqrt(tf.exp(lnvar_s))
sigma_d = tf.sqrt(tf.exp(lnvar_d))
sigma_dept = tf.sqrt(tf.exp(lnvar_dept))

mu = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
service = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

eta_s = Normal(mu=tf.zeros(n_s),
               sigma=sigma_s * tf.ones(n_s))
eta_d = Normal(mu=tf.zeros(n_d),
               sigma=sigma_d * tf.ones(n_d))
eta_dept = Normal(mu=tf.zeros(n_dept),
                  sigma=sigma_dept * tf.ones(n_dept))

yhat = tf.gather(eta_s, s_train) + \
    tf.gather(eta_d, d_train) + \
    tf.gather(eta_dept, dept_train) + \
    mu + ed.dot(service_X, service)
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

data_dict = {y: y_train, service_X: service_train}

inference = ed.KLqp(params_dict, data_dict)
inference.initialize(n_print=1, n_iter=50)

init = tf.global_variables_initializer()
init.run()

qs_mean = q_eta_s.mean()
qd_mean = q_eta_d.mean()
qdept_mean = q_eta_dept.mean()
qmu_mean = qmu.mean()
qservice_mean = qservice.mean()

y_post = ed.copy(y, {mu: qmu_mean,
                 service: qservice_mean,
                 eta_s: qs_mean,
                 eta_d: qd_mean,
                 eta_dept: qdept_mean})

service_X_test = tf.placeholder(tf.float32, [n_obs_test, 1])
yhat_test = tf.gather(qs_mean, s_test) + \
    tf.gather(qd_mean, d_test) + \
    tf.gather(qd_mean, dept_test) + \
    qmu_mean + ed.dot(service_X_test, qservice_mean)


for t in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  yhat_vals = yhat_test.eval(feed_dict={service_X_test: service_test})

  if t % inference.n_print == 0:
    plt.cla()
    plt.title("Residuals for Prediced Ratings on Test Set")
    plt.xlim(-4, 4)
    plt.ylim(0, 800)
    plt.hist(yhat_vals - y_test, 75)
    plt.draw()
    plt.pause(1.0 / 60.0)
