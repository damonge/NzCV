import numpy as np
import amplitude_model
import probabilistic_model

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

# Compute log posterior and its gradient
pm.get_log_posterior(nz*1.01)
pm.get_grad_log_posterior(nz*1.01)
