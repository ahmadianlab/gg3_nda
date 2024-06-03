# import
from inference import *
from HMM_models import *
from HMM_inference import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# define parameters ramp
beta = 2
sigma = 2

# define parameters step
m = 60
r = 5

rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T)
shmm = HMM_Step(m, r, x0, Rh, T)

N = 30
rhmm_datas = generate_N_trials(N, rhmm)
shmm_datas = generate_N_trials(N, shmm)

#################

ramp_matrix = sweep_ramp_models(rhmm_datas)
max_index_flat = np.argmax(ramp_matrix)
max_index_2d = np.unravel_index(max_index_flat, ramp_matrix.shape)
plt.scatter(max_index_2d[1], max_index_2d[0], color='red', label='Max Likelihood')
plt.scatter((np.log(sigma)-np.log(0.4)) / (np.log(4)-np.log(0.4))*M, beta/4*M, color = 'blue', label = 'True')
norm1 = mcolors.PowerNorm(gamma=3)
plt.imshow(ramp_matrix, cmap='hot', norm=norm1, interpolation='bilinear', origin='lower', aspect = 'auto')
plt.xlabel('$\sigma$')
plt.ylabel('$\\beta$')
plt.colorbar()
plt.legend()
plt.title('Likelihood heatmap for $\\beta$={:.1f}, $\sigma$={:.1f}'.format(beta,sigma))
plt.show()

step_matrix = sweep_step_models(shmm_datas)
max_index_flat = np.argmax(step_matrix)
max_index_2d = np.unravel_index(max_index_flat, step_matrix.shape)
plt.scatter(max_index_2d[1], max_index_2d[0], color='red', label='Max Likelihood')
plt.scatter((r-1)/9*10, m/T*M, color = 'blue', label = 'True')
norm2 = mcolors.PowerNorm(gamma=3)
plt.imshow(step_matrix, cmap='hot', norm=norm2, interpolation='bilinear', origin='lower', aspect = 'auto')
plt.xlabel('$r$')
plt.ylabel('$m$')
plt.colorbar()
plt.legend()
plt.title('Likelihood heatmap for $m$={:d}, $r$={:d}'.format(m,r))
plt.show()