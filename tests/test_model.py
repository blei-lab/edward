import pystan

model_code = """
data {
    int<lower=0> J; // number of schools
    real y[J]; // estimated treatment effects
    real<lower=0> sigma[J]; // s.e. of effect estimates
}
parameters {
    real mu;
    real<lower=0> tau;
    real eta[J];
}
transformed parameters {
    real theta[J];
    for (j in 1:J)
    theta[j] <- mu + tau * eta[j];
}
model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
}
"""

data = {'J': 8,
       'y': [28,  8, -3,  7, -1,  1, 18, 12],
       'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

print("The following message exists as Stan initializes an empty model.")
temp = pystan.stan(model_code=model_code, data=data, iter=1, chains=1)
