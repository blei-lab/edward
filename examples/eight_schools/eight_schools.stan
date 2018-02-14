data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real logtau;
  real theta_tilde[J];
}

model {
  mu ~ normal(0, 10); 
  logtau ~ normal(5, 1);
  theta_tilde ~ normal(0, 1);
  for (j in 1:J) {
      y[j] ~ normal(mu + exp(logtau) * theta_tilde[j], sigma[j]);
  }
}

