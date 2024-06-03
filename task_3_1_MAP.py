# import
from inference import *
from HMM_models import *
from HMM_inference import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

#parameter spaces
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

N = 30
rhmm_datas = generate_N_trials(N, rhmm)
shmm_datas = generate_N_trials(N, shmm)

ramp_l = np.exp(sweep_ramp_models(rhmm_datas))
step_l = np.exp(sweep_step_models(shmm_datas))

ramp_pos = compute_posteriori(ramp_l, ramp_priori)
step_pos = compute_posteriori(step_l, step_priori)

#######

E_sigma = np.matmul(np.sum(ramp_pos, axis = 0), sigma_space)
E_beta = np.matmul(np.sum(ramp_pos, axis = 1), beta_space)
plt.scatter((np.log(E_sigma)-np.log(0.4)) / (np.log(4)-np.log(0.4))*M, E_beta/4*M, color = 'green', label = 'E')
max_index_flat = np.argmax(ramp_pos)
max_index_2d = np.unravel_index(max_index_flat, ramp_pos.shape)
plt.scatter(max_index_2d[1], max_index_2d[0], color='red', label='MAP')
plt.scatter((np.log(sigma)-np.log(0.4)) / (np.log(4)-np.log(0.4))*M, beta/4*M, color = 'blue', label = 'True')

plt.imshow(ramp_pos, cmap='hot', norm=None, interpolation='bilinear', origin='lower', aspect = 'auto')
plt.xlabel('$\sigma$')
plt.ylabel('$\\beta$')
plt.colorbar()
plt.legend()
plt.title('Likelihood heatmap for $\\beta$={:.1f}, $\sigma$={:.1f}'.format(beta,sigma))
plt.show()

E_r = np.matmul(np.sum(step_pos, axis = 0), r_space)
E_m = np.matmul(np.sum(step_pos, axis = 1), m_space)
plt.scatter((E_r-1)/9*10, E_m/T*M, color = 'green', label = 'E')
max_index_flat = np.argmax(step_pos)
max_index_2d = np.unravel_index(max_index_flat, step_pos.shape)
plt.scatter(max_index_2d[1], max_index_2d[0], color='red', label='MAP')
plt.scatter((r-1)/9*10, m/T*M, color = 'blue', label = 'True')

plt.imshow(step_pos, cmap='hot', norm=None, interpolation='bilinear', origin='lower', aspect = 'auto')
plt.xlabel('$r$')
plt.ylabel('$m$')
plt.colorbar()
plt.legend()
plt.title('Likelihood heatmap for $m$={:d}, $r$={:d}'.format(m,r))
plt.show()