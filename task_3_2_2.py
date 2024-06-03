# import
from inference import *
from HMM_models import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import truncnorm
np.random.seed(922)


#common parameters
x0 = 0.2
Rh = 75
T = 100
K = 100

# original parameter spaces
M = 30
beta_space = np.linspace(0, 4, num=M)
sigma_space = np.exp(np.linspace(np.log(0.4), np.log(4), num=M))
m_space = np.linspace(0, T, num=M)
r_space = np.arange(10) + 1


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

N = 30
rhmm_datas = np.empty([N, T], dtype = np.int32)
shmm_datas = np.empty([N, T], dtype = np.int32)
for i in range(N):
    latent_ramp, rate_ramp, spike_ramp = rhmm.simulate()
    latent_step, rate_step, spike_step = shmm.simulate()
    rhmm_datas[i] = spike_ramp
    shmm_datas[i] = spike_step

def compute_likelihood(model, datas):
    likelihood = 0
    for d in datas: 
        ll = poisson_logpdf(d, model.lambdas, mask=None)
        likelihood += hmm_normalizer(model.initial_distribution, model.transition_matrix, ll)
    return likelihood

def sweep_ramp_models(beta_space, sigma_space, datas, normalize = False):
    l_matrix = np.empty([len(beta_space), len(sigma_space)])
    for i in range(len(beta_space)):
        for j in range(len(sigma_space)):
            model = HMM_Ramp(beta_space[i], sigma_space[j], K, x0, Rh, T)
            l_matrix[i][j] = compute_likelihood(model, datas)
    if normalize:
        return l_matrix - np.min(l_matrix), np.min(l_matrix)
    else:
        return l_matrix - np.min(l_matrix)

def sweep_step_models(m_space, r_space, datas, normalize = False):
    l_matrix = np.empty([len(m_space), len(r_space)])
    for i in range(len(m_space)):
        for j in range(len(r_space)):
            model = HMM_Step(m_space[i], r_space[j], x0, Rh, T)
            l_matrix[i][j] = compute_likelihood(model, datas)
    if normalize:
        return l_matrix - np.min(l_matrix), np.min(l_matrix)
    else:
        return l_matrix - np.min(l_matrix)

def compute_normalizer(likelihood, priori):
    mat = np.empty_like(likelihood)
    for i in range(likelihood.shape[0]):
        for j in range(likelihood.shape[1]):
            mat[i][j] = likelihood[i][j] * priori[i][j]
    norm = np.sum(mat)
    return norm

def compute_bayes_factor(datas, log = True):
    ramp_ll, ramp_coeff = sweep_ramp_models(beta_space, sigma_space, datas, normalize = True)
    step_ll, step_coeff = sweep_step_models(m_space, r_space, datas, normalize = True)
    ramp_l = np.exp(ramp_ll)
    step_l = np.exp(step_ll)
    ramp_norm = compute_normalizer(ramp_l, ramp_priori)
    step_norm = compute_normalizer(step_l, step_priori)
    if log:
        return np.log(ramp_norm / step_norm) + (ramp_coeff - step_coeff)
    else:
        return ramp_norm / step_norm * np.exp(ramp_coeff - step_coeff)
    

print(compute_bayes_factor(rhmm_datas))
print(compute_bayes_factor(shmm_datas))