# import
from inference import *
from HMM_models import *
from HMM_inference import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ramp_priori = np.ones([len(beta_space), len(sigma_space)]) / (len(beta_space)*len(sigma_space))
step_priori = np.ones([len(m_space), len(r_space)]) / (len(m_space)*len(r_space))

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
