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
M = 10
beta_space = np.linspace(0, 4, num=M)
sigma_space = np.exp(np.linspace(np.log(0.4), np.log(4), num=M))
m_space = np.linspace(0, T, num=M)
r_space = np.arange(10) + 1

# define parameters ramp
beta = 2
sigma = 2

# define parameters step
m = 60
r = 5

rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T)
shmm = HMM_Step(m, r, x0, Rh, T)

N = 25
rhmm_datas = np.empty([N, T], dtype = np.int32)
shmm_datas = np.empty([N, T], dtype = np.int32)
for i in range(N):
    latent_ramp, rate_ramp, spike_ramp = rhmm.simulate()
    latent_step, rate_step, spike_step = shmm.simulate()
    rhmm_datas[i] = spike_ramp
    shmm_datas[i] = spike_step