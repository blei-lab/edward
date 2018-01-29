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
import tensorflow as tf

from edward.models import Normal, Bernoulli
from scipy.special import expit

tf.flags.DEFINE_integer("n_students", default=50000, help="")
tf.flags.DEFINE_integer("n_questions", default=2000, help="")
tf.flags.DEFINE_integer("n_obs", default=200000, help="")

FLAGS = tf.flags.FLAGS


def build_toy_dataset(n_students, n_questions, n_obs,
                      sigma_students=1.0, sigma_questions=1.5, loc=0.0):
  student_etas = np.random.normal(0.0, sigma_students,
                                  size=n_students)
  question_etas = np.random.normal(0.0, sigma_questions,
                                   size=n_questions)

  student_ids = np.random.choice(range(n_students), n_obs)
  question_ids = np.random.choice(range(n_questions), n_obs)

  logits = student_etas[student_ids] + question_etas[question_ids] + loc
  outcomes = np.random.binomial(1, expit(logits), n_obs)

  data = pd.DataFrame({'question_id': question_ids,
                       'student_id': student_ids,
                       'outcomes': outcomes})

  return data, student_etas, question_etas


def main(_):
  ed.set_seed(42)

  # DATA
  data, true_s_etas, true_q_etas = build_toy_dataset(
      FLAGS.n_students, FLAGS.n_questions, FLAGS.n_obs)
  obs = data['outcomes'].values
  student_ids = data['student_id'].values.astype(int)
  question_ids = data['question_id'].values.astype(int)

  # MODEL
  lnvar_students = Normal(loc=0.0, scale=1.0)
  lnvar_questions = Normal(loc=0.0, scale=1.0)

  sigma_students = tf.sqrt(tf.exp(lnvar_students))
  sigma_questions = tf.sqrt(tf.exp(lnvar_questions))

  overall_mu = Normal(loc=tf.zeros(1), scale=tf.ones(1))

  student_etas = Normal(loc=0.0, scale=sigma_students,
                        sample_shape=FLAGS.n_students)
  question_etas = Normal(loc=0.0, scale=sigma_questions,
                         sample_shape=FLAGS.n_questions)

  observation_logodds = (tf.gather(student_etas, student_ids) +
                         tf.gather(question_etas, question_ids) +
                         overall_mu)
  outcomes = Bernoulli(logits=observation_logodds)

  # INFERENCE
  qstudents = Normal(
      loc=tf.get_variable("qstudents/loc", [FLAGS.n_students]),
      scale=tf.nn.softplus(
          tf.get_variable("qstudents/scale", [FLAGS.n_students])))
  qquestions = Normal(
      loc=tf.get_variable("qquestions/loc", [FLAGS.n_questions]),
      scale=tf.nn.softplus(
          tf.get_variable("qquestions/scale", [FLAGS.n_questions])))
  qlnvarstudents = Normal(
      loc=tf.get_variable("qlnvarstudents/loc", []),
      scale=tf.nn.softplus(
          tf.get_variable("qlnvarstudents/scale", [])))
  qlnvarquestions = Normal(
      loc=tf.get_variable("qlnvarquestions/loc", []),
      scale=tf.nn.softplus(
          tf.get_variable("qlnvarquestions/scale", [])))
  qmu = Normal(
      loc=tf.get_variable("qmu/loc", [1]),
      scale=tf.nn.softplus(
          tf.get_variable("qmu/scale", [1])))

  latent_vars = {
      overall_mu: qmu,
      lnvar_students: qlnvarstudents,
      lnvar_questions: qlnvarquestions,
      student_etas: qstudents,
      question_etas: qquestions
  }
  data = {outcomes: obs}
  inference = ed.KLqp(latent_vars, data)
  inference.initialize(n_print=2, n_iter=50)

  qstudents_mean = qstudents.mean()
  qquestions_mean = qquestions.mean()

  tf.global_variables_initializer().run()

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

if __name__ == "__main__":
  tf.app.run()
