from amplitude_model import *

class probabilistic_model:
    """
    Class representing an amplitude (e.g., number of galaxies N) model as a function of redshift z
    """
    def __init__(self, amplitude_model, observed_power_spectra, cov_power_spectra):
        # Initialize user input
        # Theory
        self.power_spectrum_slices = amplitude_model.power_spectrum_slices # cl_slices

        # Observations
        self.observed_power_spectra = observed_power_spectra
        self.inv_cov_power_spectra = np.linalg.inv(cov_power_spectra)

        # power_spectrum_slices [N_bin , N_bin, N_ell]
        # power_spectrum_slices [N_bin , N_bin, N_ell]
        # inv_cov_power_spectra [N_ell, N_ell]
        self.aa_grad_factor = np.einsum('kjl,lp,mnp',
                                        self.power_spectrum_slices,
                                        self.inv_cov_power_spectra,
                                        self.power_spectrum_slices)
        # power_spectrum_slices [N_bin , N_bin, N_ell]
        # observed_power_spectra [N_ell]
        # inv_cov_power_spectra [N_ell, N_ell]
        self.bb_grad_factor = np.einsum('jkl,lp,p',
                                        self.power_spectrum_slices,
                                        self.inv_cov_power_spectra,
                                        self.observed_power_spectra)


    """ Computes the (-2*)log posterior, gaussian due to the conjugate prior
    Input: self.observed_power_spectra, cl_slices, covar_cl, nz, covar_nz
    Output: (-2*)log_posterior at given data
    """
    def get_log_posterior(self, model_amplitudes):
        ##### Log-likelihood #####
        cl_theory = np.einsum('i,ijk,j', model_amplitudes, self.power_spectrum_slices, model_amplitudes)
        res = self.observed_power_spectra - cl_theory
        log_likelihood = -0.5 * np.einsum('i,ij,j',
                                          res, self.inv_cov_power_spectra, res)
        print("The log-likelihood is: " + str(log_likelihood))

        '''
        ##### Log-prior #####
        observed_amplitudes_mean = np.mean(self.observed_amplitudes)
        #sigma_nz = 0.1 * self.lowres_amplitudes + 0.02 * np.amax(self.lowres_amplitudes)
        #covar_nz = np.diag(sigma_nz ** 2)
        #inv_covar_nz = np.linalg.inv(covar_nz)
        log_prior = np.einsum('i,ij,j', (self.observed_amplitudes - observed_amplitudes_mean), self.inv_covar_amplitudes, (self.observed_amplitudes - observed_amplitudes_mean))
        print("The (-2*)log-prior is: " + str(log_prior))
        '''
        log_prior = 0 

        ##### Log-posterior #####
        log_posterior = log_likelihood + log_prior
        print("The log-posterior is: " + str(log_posterior))
        return log_posterior

    def get_grad_log_posterior(self, model_amplitudes):
        a_part = np.einsum('j,m,n,kjmn',
                           model_amplitudes,
                           model_amplitudes,
                           model_amplitudes,
                           self.aa_grad_factor)
        b_part = np.einsum('j,kj',model_amplitudes,
                           self.bb_grad_factor)

        '''
        grad_log_posterior = -2 * np.einsum('j,m,n,kja,mnb,ab', self.observed_amplitudes, self.observed_amplitudes, self.observed_amplitudes,
                                            self.power_spectra, self.power_spectra, self.inv_cov_power_spectra) \
                             + 2 * np.einsum('j,kja,ab,b', self.observed_amplitudes, self.power_spectra, self.inv_cov_power_spectra, self.observed_power_spectra) \
                             + np.einsum('kj,j', self.inv_covar_amplitudes, (self.observed_amplitudes - np.mean(self.observed_amplitudes)))
        return grad_log_posterior
        '''
        return a_part - b_part
        
if __name__ == '__main__':
    ### Use case example ###

    # User input: 3 axes
    z_lores = np.linspace(0, 2, 11)
    z_hires = np.linspace(0, 2.5, 1280)
    ells = np.linspace(100, 2000, 100)

    # Build model
    model = amplitude_model(z_hires, z_lores, ells)
    exit(1)
    ##### Load observed data #####
    with np.load("nz_data.npz") as nz_data:
        nz, covar_nz, z_low, z_high = nz_data['nz'], nz_data['nz_covar'], nz_data['z_edges_lo'], nz_data['z_edges_hi']

    with np.load("cls_data.npz") as cls_data:
        ls, cl_data, covar_cl = cls_data['ls'], cls_data['cls'], cls_data['cs_covar']

    # Build probabilistic model
    pm = probabilistic_model(model, nz, np.linalg.inv(covar_nz), cl_data, np.linalg.inv(covar_cl))
    pm.get_log_posterior()
    print(pm.get_grad_log_posterior())

    # Plot
    model.plot_amplitudes()
    model.plot_power_spectra()
