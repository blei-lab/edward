library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

schools_dat <- list(J = 8, 
                    y = c(28,  8, -3,  7, -1,  1, 18, 12),
                    sigma = c(15, 10, 16, 11,  9, 11, 10, 18))


fit <- stan(file = 'eight_schools.stan', data = schools_dat,iter = 100000, chains = 4)
print(fit)
