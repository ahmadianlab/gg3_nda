# import
from inference import *
from HMM_models import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

#common parameters
x0 = 0.2
Rh = 75
T = 100
K = 100
<<<<<<< HEAD
gamma = 5
=======
>>>>>>> main

#parameter spaces
M = 10
beta_space = np.linspace(0, 4, num=M)
sigma_space = np.exp(np.linspace(np.log(0.4), np.log(4), num=M))
m_space = np.linspace(0, T, num=M)
r_space = np.arange(10) + 1

<<<<<<< HEAD
ramp_priori = np.ones([len(beta_space), len(sigma_space)]) / (len(beta_space)*len(sigma_space))
step_priori = np.ones([len(m_space), len(r_space)]) / (len(m_space)*len(r_space))


=======
>>>>>>> main
# define parameters ramp
beta = 2
sigma = 2

# define parameters step
m = 60
r = 5

<<<<<<< HEAD
rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T, isi_gamma_shape = gamma)
shmm = HMM_Step(m, r, x0, Rh, T, isi_gamma_shape = gamma)
=======
rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T)
shmm = HMM_Step(m, r, x0, Rh, T)
>>>>>>> main

N = 25
rhmm_datas = np.empty([N, T], dtype = np.int32)
shmm_datas = np.empty([N, T], dtype = np.int32)
for i in range(N):
    latent_ramp, rate_ramp, spike_ramp = rhmm.simulate()
    latent_step, rate_step, spike_step = shmm.simulate()
    rhmm_datas[i] = spike_ramp
<<<<<<< HEAD
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
=======
    shmm_datas[i] = spike_step
>>>>>>> main
