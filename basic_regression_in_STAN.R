
#   -----------------------------------------------------------------------
# A basic linear regression in STAN
#   -----------------------------------------------------------------------

# following: 
# STAN users guide: https://mc-stan.org/docs/2_19/stan-users-guide/linear-regression.html
# https://ourcodingclub.github.io/2018/04/17/stan-intro.html


# initialization ----------------------------------------------------------

library(rstan)
options(mc.cores = parallel::detectCores()) # set up to estimate your model in parallel
rstan_options(auto_write = TRUE) # automatically save a bare version of a compiled Stan program to the hard disk so that it does not need to be recompiled
library(dplyr)

# model -------------------------------------------------------------------

write("// Stan model for simple linear regression

data {
 int < lower = 0 > N; // sample size
 int < lower = 0 > K; // number of predictors
 matrix[N, K] x; // predictor matrix
 vector[N] y; // response
 int < lower = 0 > N_new; // predicted sample size
 matrix[N, K] x_new; // prediction matrix
}

parameters {
 real a; // intercept
 vector[K] b; // predictor coefficients
 real < lower = 0 > sigma; // error (scaled)
}

model {
 a ~ normal(0,1e6);
 b[K] ~ normal(0,1e6);
 y ~ normal(a + x*b, sigma);
}

generated quantities {
  vector[N_new] y_new;
  for (n in 1:N_new)
    y_new[n] = normal_rng(a + x_new[n] * b, sigma);
} // The posterior predictive distribution",

"stan_lm.stan")

stanc("stan_lm.stan") # compile model


# generate data -----------------------------------------------------------

x <- as.matrix(seq(-1,1,length=50))
y <- rnorm(50,10+20*x,5)
plot(x,y)

stan_data <- list(N = 50, K=ncol(x), x = x, y = y, x_new=x, N_new=50)

# run model ---------------------------------------------------------------

fit <- stan(file="stan_lm.stan", data=stan_data, warmup=500, iter=1000, chains=3, cores=3, thin=1)


# examine outputs ---------------------------------------------------------

fit
print(fit, probs = c(0.10, 0.5, 0.9), pars=c('a','b','sigma'))

posterior <- extract(fit)
str(posterior)
for (i in 1:1500) {
  abline(posterior$a[i], posterior$b[i], col = rgb(190,190,190,100,max=255), lty = 1)
}
abline(mean(posterior$a),mean(posterior$b))

post <- as.data.frame(fit)
post %>% select(a, 'b[1]', sigma) %>% cor() # correlation matrix
(coef_mean <- post %>% select(a, 'b[1]', sigma) %>% summarise_all(mean) %>% as.numeric) # parameter means


# convergence diagnostics -------------------------------------------------

plot(posterior$a, type = "l") # by hand
traceplot(fit, pars='a') # built in function
stan_dens(fit, pars='a')
# plot(fit)


# posterior checks --------------------------------------------------------

y_rep <- as.matrix(fit, pars = "y_new")
dim(y_rep)
library(bayesplot)
ppc_dens_overlay(y, y_rep[1:200, ]) # comparing density of y with densities of y over 200 posterior draws.
ppc_stat(y = y, yrep = y_rep, stat = "mean") # compare estimates of summary statistics
ppc_scatter_avg(y = y, yrep = y_rep) # observed vs predicted with 1:1 line
