import numpy as np
import matplotlib
matplotlib.use('TkAgg') # need this for unknown reason on macosX
import matplotlib.pyplot as plt
import pymc3 as pm

# initialize random number generator
np.random.seed(123)

# true parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

# pymc3 code
basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

    # draw 500 posterior samples
    trace = pm.sample(500)

pm.traceplot(trace)
plt.show()
# plot
#fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))

#axes[0].scatter(X1, Y)
#axes[1].scatter(X2, Y)
#axes[0].set_ylabel('Y')
#axes[0].set_xlabel('X1')
#axes[1].set_xlabel('X2')

#plt.show()