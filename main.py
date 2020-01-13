import numpy as np
from amplitude_model import *
from probabilistic_model import *
from pyhmc import hmc

# User input: 3 axes
z_lores = np.linspace(0, 2, 11)
z_hires = np.linspace(0, 2.5, 1280)
ells = np.linspace(100, 2000, 100)

# Build amplitude model based on 3 axes
model = amplitude_model(z_hires, z_lores, ells)

# Load observed data
with np.load("nz_data.npz") as nz_data:
    nz, covar_nz, z_low, z_high = nz_data['nz'], nz_data['nz_covar'], nz_data['z_edges_lo'], nz_data['z_edges_hi']
with np.load("cls_data.npz") as cls_data:
    ls, cl_data, covar_cl = cls_data['ls'], cls_data['cls'], cls_data['cs_covar']

# Build probabilistic model based on amplitude model and observed data
pm = probabilistic_model(model, nz, covar_nz, cl_data, np.linalg.inv(covar_cl))

# The PyHMC package requires a function returning both the logprob and its gradient
def logposterior_and_grad(model_amplitudes):
    return pm.get_log_posterior(model_amplitudes), pm.get_grad_log_posterior(model_amplitudes)

# Draw samples from the log posterior using hamiltonian monte carlo
print("A sensible sample would be..."+ str(nz))
initial_position = np.random.rand(10)
nb_samples = 10
print("The initial position is..."+str(initial_position))
samples = hmc(logposterior_and_grad, x0=initial_position, n_samples=10, n_steps=10, epsilon=0.001)
print("Samples drawn: \n"+str(samples))
