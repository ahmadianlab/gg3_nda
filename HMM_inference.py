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

#parameter spaces
M = 30
beta_space = np.linspace(0, 4, num=M)
sigma_space = np.exp(np.linspace(np.log(0.4), np.log(4), num=M))
m_space = np.linspace(0, T, num=M)
r_space = np.arange(10) + 1

def generate_N_trials(N, model):

    datas = np.empty([N, T], dtype = np.int32)
    for i in range(N):
        latent, rate, spike = model.simulate()
        datas[i] = spike
    return datas

def compute_ll(model, datas):
    likelihood = 0
    for d in datas: 
        ll = poisson_logpdf(d, model.lambdas, mask=None)
        likelihood += hmm_normalizer(model.initial_distribution, model.transition_matrix, ll)
    return likelihood

def sweep_ramp_models(datas, normalize = False):
    l_matrix = np.empty([len(beta_space), len(sigma_space)])
    for i in range(len(beta_space)):
        for j in range(len(sigma_space)):
            model = HMM_Ramp(beta_space[i], sigma_space[j], K, x0, Rh, T)
            l_matrix[i][j] = compute_ll(model, datas)
    if normalize:
        return l_matrix - np.max(l_matrix), np.max(l_matrix)
    else:
        return l_matrix - np.max(l_matrix)

def sweep_step_models(datas, normalize = False):
    l_matrix = np.empty([len(m_space), len(r_space)])
    for i in range(len(m_space)):
        for j in range(len(r_space)):
            model = HMM_Step(m_space[i], r_space[j], x0, Rh, T)
            l_matrix[i][j] = compute_ll(model, datas)
    if normalize:
        return l_matrix - np.max(l_matrix), np.max(l_matrix)
    else:
        return l_matrix - np.max(l_matrix)

def compute_normalizer(likelihood, priori):
    mat = np.empty_like(likelihood)
    for i in range(likelihood.shape[0]):
        for j in range(likelihood.shape[1]):
            mat[i][j] = likelihood[i][j] * priori[i][j]
    norm = np.sum(mat)
    return norm

def compute_bayes_factor(datas, ramp_priori, step_priori, log = True):
    ramp_ll, ramp_coeff = sweep_ramp_models(datas, normalize = True)
    step_ll, step_coeff = sweep_step_models(datas, normalize = True)
    ramp_l = np.exp(ramp_ll)
    step_l = np.exp(step_ll)
    ramp_norm = compute_normalizer(ramp_l, ramp_priori)
    step_norm = compute_normalizer(step_l, step_priori)
    if log:
        return np.log(ramp_norm / step_norm) + (ramp_coeff - step_coeff)
    else:
        return ramp_norm / step_norm * np.exp(ramp_coeff - step_coeff)

def compute_posteriori(likelihood, priori):
    mat = np.empty_like(likelihood)
    for i in range(likelihood.shape[0]):
        for j in range(likelihood.shape[1]):
            mat[i][j] = likelihood[i][j] * priori[i][j]
    norm = np.sum(mat)
    mat = mat / norm
    return mat