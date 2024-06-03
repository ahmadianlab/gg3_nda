# import
from inference import *
from HMM_models import *
from HMM_inference import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import truncnorm
np.random.seed(922)


# Truncated Gaussian prior
beta_center = beta_space[0]
sigma_center = (sigma_space[0] + sigma_space[-1]) / 2
m_center = m_space[0]
r_center = (r_space[0]+r_space[-1]) / 2

fraction = 0.1
beta_std = fraction * (beta_space[-1] - beta_space[0])
sigma_std = fraction * (sigma_space[-1] - sigma_space[0])
m_std = fraction * (m_space[-1] - m_space[0])
r_std = fraction * (r_space[-1] - r_space[0])

def truncated_gaussian_prior(mean, std_dev, lower, upper, size=1):
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    return truncnorm(a, b, loc=mean, scale=std_dev).rvs(size)

beta_prior = truncated_gaussian_prior(beta_center, beta_std, beta_space[0], beta_space[-1], size = len(beta_space))
sigma_prior = truncated_gaussian_prior(sigma_center, sigma_std, sigma_space[0], sigma_space[-1], size = len(sigma_space))
m_prior = truncated_gaussian_prior(m_center, m_std, m_space[0], m_space[-1], size = len(m_space))
r_prior = truncated_gaussian_prior(r_center, r_std, r_space[0], r_space[-1], size = len(r_space))

for p in [beta_prior, sigma_prior, m_prior, r_prior]:
    p /= np.sum(p)

ramp_priori = np.outer(beta_prior, sigma_prior)
step_priori = np.outer(m_prior, r_prior)


# define parameters ramp
beta = 2
sigma = 2

# define parameters step
m = 60
r = 5

rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T)
shmm = HMM_Step(m, r, x0, Rh, T)

N = 5
rhmm_datas = generate_N_trials(N, rhmm)
shmm_datas = generate_N_trials(N, shmm)

print(compute_bayes_factor(rhmm_datas, ramp_priori, step_priori))
print(compute_bayes_factor(shmm_datas, ramp_priori, step_priori))
