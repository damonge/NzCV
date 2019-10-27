import os
import itertools
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

""" Gaussian N(z)
Input: z
Output: evaluation of z
"""
def nz_model(z):
    sigma=0.2
    mean=1.0
    return np.exp(-0.5*((z-mean)/sigma)**2)/np.sqrt(2*np.pi*sigma**2)

""" Heaviside basis function
Input: z, z_l, z_h
Output: evaluations of heaviside at z
"""
def window_step(z,z_l,z_h):
    return (np.heaviside(z-z_l,0.)-np.heaviside(z-z_h,0.0))/(z_h-z_l)

""" Compute cl with Weak lensing tracers according to cosmological parameters
Input: cosmo, nz1, nz2, ells
Output: cl
"""
def get_cl(cosmo,nz1,nz2,ells):
    # This computes the power spectrum between two redshift distributions
    t1=ccl.WeakLensingTracer(cosmo,nz1)
    t2=ccl.WeakLensingTracer(cosmo,nz2)
    return ccl.angular_cl(cosmo,t1,t2,ells)

""" Make high resolution redshift axis z
Input: z_start, z_end, nz_size
Output: z
"""
def make_z(z_start, z_end, nz_size):
    return np.linspace(z_start,z_end+0.5,int(nz_size*(1+0.5/(z_end-z_start))))

""" Make nz_slices
Input: z, z_low, z_high
Output: nz_slices
"""
def make_nz_slices(z, z_low, z_high):
    return np.array([window_step(z,z_l,z_h) for z_l,z_h in zip(z_low,z_high)])

""" Make cl slices
Input: n_zsam, ell_size, ells, cosmo_params, z, nz_slices
Output: cl_slices
"""
def make_cl_slices(n_zsam, ell_size, ells, cosmo_params, z, nz_slices):
    cosmo = ccl.Cosmology(Omega_b=cosmo_params[0],
                          Omega_c=cosmo_params[1],
                          sigma8=cosmo_params[2],
                          h=cosmo_params[3],
                          n_s=cosmo_params[4],
                          Omega_k=cosmo_params[5])
    cls_slices = np.zeros([n_zsam, n_zsam, ell_size])
    for i1 in range(n_zsam):
        for i2 in range(i1, n_zsam):
            cls_slices[i1, i2, :] = get_cl(cosmo, (z, nz_slices[i1]), (z, nz_slices[i2]), ells)
            if i1 != i2:
                cls_slices[i2, i1, :] = cls_slices[i1, i2, :]
    return cls_slices

""" Make fake data and saves it to two npz files
Input: z, cosmo_params, z_start, z_end, n_zsam, ells
Output: For plotting purposes only: nz_lores, z_mid, nz_lores_data, sigma_nz, nz_steps, z_hi, z_lo
"""
def make_fake_data(z, cosmo_params, z_start, z_end, n_zsam, ells):
    # Array containing the edges of each redshift slice
    z_lores = np.linspace(z_start, z_end, n_zsam + 1)
    # Low edges
    z_lo = z_lores[:-1]
    # High edges
    z_hi = z_lores[1:]
    # Mid-point
    z_mid = 0.5 * (z_lo + z_hi)
    # In each slice we just take the value of
    # the Gaussian above at the centre of the bin.
    nz_lores = nz_model(z_mid)

    nz_lores /= np.sum(nz_lores)

    nz_slices = make_nz_slices(z, z_lo, z_hi)
    # This is our model for the redshift distribution
    nz_steps = np.einsum('j,ji', nz_lores, nz_slices)

    # Power spectrum for the model redshift distribution
    cosmo = ccl.Cosmology(Omega_b=cosmo_params[0],
                          Omega_c=cosmo_params[1],
                          sigma8=cosmo_params[2],
                          h=cosmo_params[3],
                          n_s=cosmo_params[4],
                          Omega_k=cosmo_params[5])
    cls_steps = get_cl(cosmo,
                       (z, nz_steps),
                       (z, nz_steps),
                       ells)
    sigma_gamma = 0.28
    ndens = 10.
    nls = sigma_gamma ** 2 * \
          np.ones_like(cls_steps) / \
          (ndens * (180 * 60 / np.pi) ** 2)

    # Covariance matrix (assuming 10% sky fraction)
    D_ell = (ell_end - ell_start) / ell_size
    fsky = 0.1
    covar_cl = np.diag((cls_steps + nls) ** 2 / ((ells + 0.5) * fsky * D_ell))

    # Now let's generate some fake power spectrum data
    cls_data = np.random.multivariate_normal(cls_steps, covar_cl)
    np.savez('cls_data',
             ls=ells,
             cls=cls_data,
             cs_covar=covar_cl)

    # Now let's do the same thing for the redshift distribution
    # Let's say that we have 10% errors + some offset so we have some
    # extra constant noise (so the errors aren't 0 where N(z) is 0).
    sigma_nz = 0.1 * nz_lores + 0.02 * np.amax(nz_lores)
    covar_nz = np.diag(sigma_nz ** 2)
    nz_lores_data = np.random.multivariate_normal(nz_lores, covar_nz)
    np.savez('nz_data',
             nz=nz_lores_data,
             nz_covar=covar_nz,
             z_edges_lo=z_lo,
             z_edges_hi=z_hi)
    return [nz_lores, z_mid, nz_lores_data, sigma_nz, nz_steps, z_hi, z_lo]

""" Plot distribution data
Input: nz_lores, z_mid, nz_lores_data, sigma_nz, nz_steps, z_hi, z_lo
Output: None, shows plot
"""
def plot_data(nz_lores, z_mid, nz_lores_data, sigma_nz, nz_steps, z_hi, z_lo):
    # Now sandwich with N(z) amplitudes
    cls_sandwich = np.einsum('i,ijk,j', nz_lores, cl_slices, nz_lores)
    # Compare high-resolution N(z) and low-resolution slicing
    nz_smooth = nz_model(z)
    plt.figure()
    for n in nz_slices:
        plt.plot(z, n, 'k-', lw=1)
    plt.errorbar(z_mid,
                 nz_lores_data / (z_hi - z_lo),
                 yerr=sigma_nz / (z_hi - z_lo),
                 fmt='g.')
    plt.plot(z, nz_smooth, 'r-')
    plt.plot(z, nz_steps, 'b-')
    plt.show()
""" Plot power spectra
Input: cls_steps, cls_sandwich, nls
Output: None, shows plot
"""
def plot_power_spectra(cls_steps, cls_sandwich, nls):
    # Power spectra
    plt.figure()
    plt.plot(ells, cls_steps, 'b-')
    plt.plot(ells, cls_sandwich, 'r--')
    plt.errorbar(ells, cls_data,
                 yerr=np.sqrt(np.diag(covar_cl)),
                 fmt='g.')
    plt.plot(ells, nls, 'k--', lw=1)
    plt.loglog()
    plt.show()

""" Computes the (-2*)log posterior, gaussian due to the conjugate prior
Input: cl_data, cl_slices, covar_cl, nz, covar_nz
Output: (-2*)log_posterior at given data
"""
def evaluate_log_posterior(cl_data, cl_slices, covar_cl, nz, covar_nz):
    ##### Log-likelihood #####
    z_range = range(nz.shape[0])
    l_range = range(cl_data.shape[0])
    inv_covar_cl = np.linalg.inv(covar_cl)
    log_likelihood = 0
    for i, j, k, l, m, n in itertools.product(z_range, z_range, l_range, l_range, z_range, z_range):
        if(inv_covar_cl[k,l] != 0.):
            left_term = (nz[i] * nz[j] * cl_slices[i, j, k]) - cl_data[k]
            right_term = (nz[m] * nz[n] * cl_slices[m, n, l]) - cl_data[l]
            log_likelihood += left_term * inv_covar_cl[k,l] * right_term
    print("The (-2*)log-likelihood is: "+str(log_likelihood))

    ##### Log-prior #####
    nz_mean = np.mean(nz)
    inv_covar_nz = np.linalg.inv(covar_nz)
    log_prior = np.einsum('i,ij,j', (nz-nz_mean), inv_covar_nz, (nz-nz_mean))
    print("The (-2*)log-prior is: "+str(log_prior))

    ##### Log-posterior #####
    log_posterior = log_likelihood + log_prior
    print("The (-2*)log-posterior is: "+str(log_posterior))
    return log_posterior

#################################################################################
################################## MAIN METHOD ##################################
#################################################################################
if __name__ == '__main__':

    ##### User Input #####
    ell_size = 100
    ell_start, ell_end = 100., 2000.

    nz_size = 1024
    z_start, z_end = 0, 2
    n_zsam = 10

    ##### Cosmological parameters Omega_b, Omega_c, sigma8, h, n, Omega_k #####
    cosmo_params = [0.05, 0.25, 0.8, 0.67, 0.96, 0]

    ##### Build z and ells from user input #####
    z = make_z(z_start, z_end, nz_size)
    ells = np.linspace(ell_start, ell_end, ell_size)

    ##### If there is no data, create some #####
    if not(os.path.isfile('nz_data.npz') and os.path.isfile('cls_data.npz')):
        plotting_params = make_fake_data(z, cosmo_params, z_start, z_end, n_zsam, ells)

    ##### Load Data #####
    with np.load("nz_data.npz") as nz_data:
        nz, covar_nz, z_low, z_high = nz_data['nz'], nz_data['nz_covar'], nz_data['z_edges_lo'], nz_data['z_edges_hi']

    with np.load("cls_data.npz") as cls_data:
        ls, cl_data, covar_cl = cls_data['ls'], cls_data['cls'], cls_data['cs_covar']

    ##### Build Model and Compute log-posterior according to data and cosmological parameters #####
    nz_slices = make_nz_slices(z, z_low, z_high)
    cl_slices = make_cl_slices(n_zsam, ell_size, ells, cosmo_params, z, nz_slices)
    log_posterior = evaluate_log_posterior(cl_data, cl_slices, covar_cl, nz, covar_nz)

    ##### (Optional) Plot some fake data #####
    #plot_data(*make_fake_data(z, cosmo_params, z_start, z_end, n_zsam, ells))