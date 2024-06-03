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
gamma = 5

t = 1000

# define parameters ramp
beta = 2
sigma = 2

# define parameters step
m = 60
r = 5

rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T, isi_gamma_shape = gamma)
shmm = HMM_Step(m, r, x0, Rh, T, isi_gamma_shape = gamma)

def trial_average(model, iterations, t, N):
    
    bin = np.zeros(t)
    for i in range(iterations):
        latent, rate, spikes = model.simulate()
        bin += spikes[0]
    bin  = bin / iterations
    
    bin = np.convolve(bin, np.ones(N)/N, mode='valid')
    return bin * t

model = rhmm
bin = trial_average(model, 2500, t, 50)
spike_times = np.linspace(0, 1, num = bin.shape[0], endpoint = False)

plt.plot(spike_times, bin, label = 'PSTH over 2500 samples')
plt.title('PSTH of step model  '+'m ='+str(m)+'  r='+str(r))
#plt.title('PSTH of ramp model  ' + '$\\beta$=' + str(bet) + '  $\sigma$=' + str(sig))
plt.xlabel('time (s)   ' + 't=' + str(t))
plt.legend()
plt.show()
