#!/usr/bin/env python
"""Bayesian Item Response Theory (IRT) Mixed Effects Model
using variational inference.

Simulates data and fits y ~ 1 + (1|student) + (1|question)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf

from edward.models import Normal, Bernoulli


def make_toy_data(n_students, n_questions, n_obs,
                  sigma_students=1.0, sigma_questions=1.5, mu=0.0):
  student_etas = np.random.normal(0.0, sigma_students,
                                  size=n_students)
  question_etas = np.random.normal(0.0, sigma_questions,
                                   size=n_questions)

  student_ids = np.random.choice(range(n_students), n_obs)
  question_ids = np.random.choice(range(n_questions), n_obs)

  logits = student_etas[student_ids] + question_etas[question_ids] + mu
  outcomes = np.random.binomial(1, sp.special.expit(logits), n_obs)

  data = pd.DataFrame({'question_id': question_ids,
                       'student_id': student_ids,
                       'outcomes': outcomes})

  return data, student_etas, question_etas


ed.set_seed(42)

n_students = 50000
n_questions = 2000
n_obs = 200000

# DATA
data, true_s_etas, true_q_etas = make_toy_data(n_students, n_questions, n_obs)
obs = data['outcomes'].values
student_ids = data['student_id'].values.astype(int)
question_ids = data['question_id'].values.astype(int)

# MODEL
lnvar_students = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
lnvar_questions = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

sigma_students = tf.sqrt(tf.exp(lnvar_students))
sigma_questions = tf.sqrt(tf.exp(lnvar_questions))

overall_mu = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

student_etas = Normal(mu=tf.zeros(n_students),
                      sigma=sigma_students * tf.ones(n_students))
question_etas = Normal(mu=tf.zeros(n_questions),
                       sigma=sigma_questions * tf.ones(n_questions))

observation_logodds = tf.gather(student_etas, student_ids) + \
    tf.gather(question_etas, question_ids) + \
    overall_mu
outcomes = Bernoulli(logits=observation_logodds)


# INFERENCE
def make_normal(n):
  var = Normal(
      mu=tf.Variable(tf.random_normal([n])),
      sigma=tf.nn.softplus(tf.Variable(tf.random_normal([n]))))
  return var


qstudents = make_normal(n_students)
qquestions = make_normal(n_questions)
qlnvarstudents = make_normal(1)
qlnvarquestions = make_normal(1)
qmu = make_normal(1)

latent_dict = {
    overall_mu: qmu,
    lnvar_students: qlnvarstudents,
    lnvar_questions: qlnvarquestions,
    student_etas: qstudents,
    question_etas: qquestions
}
data_dict = {outcomes: obs}

inference = ed.KLqp(latent_dict, data_dict)
inference.initialize(n_print=2, n_iter=50)

qstudents_mean = qstudents.mean()
qquestions_mean = qquestions.mean()

init = tf.global_variables_initializer()
init.run()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_ylim([-3.0, 3.0])
ax2.set_ylim([-3.0, 3.0])
ax1.set_xlim([-3.0, 3.0])
ax2.set_xlim([-3.0, 3.0])

for t in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

  if t % inference.n_print == 0:
    # CRITICISM
    ax1.clear()
    ax2.clear()
    ax1.set_ylim([-3.0, 3.0])
    ax2.set_ylim([-3.0, 3.0])
    ax1.set_xlim([-3.0, 3.0])
    ax2.set_xlim([-3.0, 3.0])

    ax1.set_title('Student Intercepts')
    ax2.set_title('Question Intercepts')
    ax1.set_xlabel('True Student Random Intercepts')
    ax1.set_ylabel('Estimated Student Random Intercepts')
    ax2.set_xlabel('True Question Random Intercepts')
    ax2.set_ylabel('Estimated Question Random Intercepts')

    ax1.scatter(true_s_etas, qstudents_mean.eval(), s=0.05)
    ax2.scatter(true_q_etas, qquestions_mean.eval(), s=0.05)
    plt.draw()
    plt.pause(2.0 / 60.0)
