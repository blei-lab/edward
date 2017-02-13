#!/usr/bin/env python
"""Bayesian IRT Mixed Effects Model using variational inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf

import matplotlib.pyplot as plt

from edward.models import Normal, Bernoulli


data = pd.read_csv('/Users/pfoley/irtexample.csv')

ed.set_seed(42)

n_students = 1000
n_questions = 100
n_obs = 10000

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
                      sigma=sigma_students*tf.ones(n_students))
question_etas = Normal(mu=tf.zeros(n_questions),
                       sigma=sigma_questions*tf.ones(n_questions))

observation_logodds = tf.gather(student_etas, student_ids) + \
                      tf.gather(question_etas, question_ids) + \
                      overall_mu
outcomes = Bernoulli(logits=observation_logodds)

# INFERENCE
T = 5000
qstudents = Normal(mu=tf.Variable(tf.random_normal([n_students])),
                   sigma=tf.nn.softplus(tf.Variable(tf.random_normal([n_students]))))
qquestions = Normal(mu=tf.Variable(tf.random_normal([n_questions])),
                    sigma=tf.nn.softplus(tf.Variable(tf.random_normal([n_questions]))))
qlnvarstudents = Normal(mu=tf.Variable(tf.random_normal([1])),
                        sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
qlnvarquestions = Normal(mu=tf.Variable(tf.random_normal([1])),
                         sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

qmu = Normal(mu=tf.Variable(tf.random_normal([1])),
             sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))


params_dict = {
    overall_mu: qmu,
    lnvar_students: qlnvarstudents,
    lnvar_questions: qlnvarquestions,
    student_etas: qstudents,
    question_etas: qquestions
}

data_dict = {outcomes: obs}

inference = ed.KLqp(params_dict, data_dict)
inference.initialize(n_print=20, n_iter=200)

#inference.run(n_iter=1000)

init = tf.global_variables_initializer()
init.run()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


for t in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    if t % inference.n_print == 0:

        # CRITICISM
        student_etas_post = qstudents.mean().eval()
        question_etas_post = qquestions.mean().eval()

        plt.cla()
        ax1.cla()
        ax2.cla()
        ax1.scatter(true_s_etas, student_etas_post)
        ax2.scatter(true_q_etas, question_etas_post)
        plt.draw()
        plt.pause(1.0 / 60.0)

