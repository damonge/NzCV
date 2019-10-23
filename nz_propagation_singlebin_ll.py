import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

z_ini=0.
z_end=2.
nz_hires=1024

n_zsam=10

# For simplicity let's use a Gaussian N(z)
def nz_model(z):
    sigma=0.2
    mean=1.0
    return np.exp(-0.5*((z-mean)/sigma)**2)/np.sqrt(2*np.pi*sigma**2)

# High-resolution array of redshifts
z_hires=np.linspace(z_ini,z_end+0.5,int(nz_hires*(1+0.5/(z_end-z_ini))))

# Array containing the edges of each redshift slice
z_lores=np.linspace(z_ini,z_end,n_zsam+1)
# Low edges
z_lo=z_lores[:-1]
# High edges
z_hi=z_lores[1:]
# Mid-point
z_mid=0.5*(z_lo+z_hi)

def window_step(z,z_l,z_h):
    return (np.heaviside(z-z_l,0.)-np.heaviside(z-z_h,0.0))/(z_h-z_l)


# Now lets generate redshift distributions for each slice
nz_slices=np.array([window_step(z_hires,z_l,z_h)
                    for z_l,z_h in zip(z_lo,z_hi)])

# In each slice we just take the value of
# the Gaussian above at the centre of the bin.
nz_lores=nz_model(z_mid)
nz_lores/=np.sum(nz_lores)

# This is our model for the redshift distribution
nz_steps=np.einsum('j,ji', nz_lores, nz_slices)


# Alright, power spectra
N_ell = 100
ell_ini = 100.
ell_end = 2000.
ells = np.linspace(ell_ini, ell_end, N_ell)
csm=ccl.Cosmology(Omega_b=0.05,
                  Omega_c=0.25,
                  sigma8=0.8,
                  h=0.67,
                  n_s=0.96,
                  Omega_k=0)

def get_cl(cosmo,nz1,nz2):
    # This computes the power spectrum between two redshift distributions
    t1=ccl.WeakLensingTracer(cosmo,nz1)
    t2=ccl.WeakLensingTracer(cosmo,nz2)
    return ccl.angular_cl(cosmo,t1,t2,ells)


# Power spectrum for the model redshift distribution
cls_steps=get_cl(csm,
                 (z_hires,nz_steps),
                 (z_hires,nz_steps))

# Power spectra for each pair of slices
cls_slices=np.zeros([n_zsam,n_zsam,N_ell])
for i1 in range(n_zsam):
    for i2 in range(i1,n_zsam):
        print(i1,i2)
        cls_slices[i1,i2,:]=get_cl(csm,
                                   (z_hires,nz_slices[i1]),
                                   (z_hires,nz_slices[i2]))
        if i1!=i2:
            cls_slices[i2,i1,:]=cls_slices[i1,i2,:]


# Now sandwich with N(z) amplitudes
cls_sandwich=np.einsum('i,ijk,j',nz_lores,cls_slices,nz_lores)


# Let's now make some fake data
# Noise power spectrum, assuming 10 gals/arcmin^2
sigma_gamma = 0.28
ndens = 10.
nls = sigma_gamma**2 * \
      np.ones_like(cls_steps) / \
      (ndens * (180 * 60 / np.pi)**2)
# Covariance matrix (assuming 10% sky fraction)
D_ell = (ell_end - ell_ini) / N_ell
fsky = 0.1
covar_cl = np.diag((cls_steps + nls)**2/((ells + 0.5) * fsky * D_ell))
# Now let's generate some fake power spectrum data
cls_data = np.random.multivariate_normal(cls_steps, covar_cl)
np.savez('cls_data',
         cls=cls_data,
         cs_covar=covar_cl)

# Now let's do the same thing for the redshift distribution
# Let's say that we have 10% errors + some offset so we have some
# extra constant noise (so the errors aren't 0 where N(z) is 0).
sigma_nz = 0.1 * nz_lores + 0.02 * np.amax(nz_lores)
covar_nz = np.diag(sigma_nz**2)
nz_lores_data = np.random.multivariate_normal(nz_lores, covar_nz)
np.savez('nz_data',
         nz=nz_lores_data,
         nz_covar=covar_nz,
         z_edges_lo=z_lo,
         z_edges_hi=z_hi)

# Compare high-resolution N(z) and low-resolution slicing
nz_smooth=nz_model(z_hires)
plt.figure()
for n in nz_slices:
    plt.plot(z_hires,n,'k-',lw=1)
plt.errorbar(z_mid,
             nz_lores_data/(z_hi-z_lo),
             yerr = sigma_nz/(z_hi-z_lo),
             fmt='g.')
plt.plot(z_hires,nz_smooth,'r-')
plt.plot(z_hires,nz_steps,'b-')

# Power spectra
plt.figure()
plt.plot(ells,cls_steps,'b-')
plt.plot(ells,cls_sandwich,'r--')
plt.errorbar(ells,cls_data,
             yerr=np.sqrt(np.diag(covar_cl)),
             fmt='g.')
plt.plot(ells,nls,'k--',lw=1)
plt.loglog()

plt.show()
