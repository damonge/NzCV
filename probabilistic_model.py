import numpy as np

class probabilistic_model:
    """
    Class representing an amplitude (e.g., number of galaxies N) model as a function of redshift z
    """
    def __init__(self, amplitude_model, nz, covar_nz, observed_power_spectra, cov_power_spectra):
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
        self.observed_amplitudes = nz
        self.inv_cov_amplitudes = np.linalg.inv(covar_nz)


    def get_log_posterior(self, model_amplitudes):
        """
        Compute the log posterior
        Input: model_amplitudes
        Output: log_posterior at given data
        """
        ##### Log-likelihood #####
        cl_theory = np.einsum('i,ijk,j', model_amplitudes, self.power_spectrum_slices, model_amplitudes)
        res = self.observed_power_spectra - cl_theory
        log_likelihood = -0.5 * np.einsum('i,ij,j',
                                          res, self.inv_cov_power_spectra, res)
        #print("The log-likelihood is: " + str(log_likelihood))


        ##### Log-prior #####
        d_amplitude = model_amplitudes - self.observed_amplitudes
        log_prior = -0.5 * np.einsum('i,ij,j', d_amplitude, self.inv_cov_amplitudes, d_amplitude)
        #print("The log-prior is: " + str(log_prior))

        ##### Log-posterior #####
        log_posterior = log_likelihood + log_prior
        #print("The log-posterior is: " + str(log_posterior))
        return log_posterior


    def get_grad_log_posterior(self, model_amplitudes):
        """
        Compute the gradient of the log posterior
        Input: model_amplitudes
        Output: gradient of the log_posterior at given data
        """
        a_part = np.einsum('j,m,n,kjmn',
                           model_amplitudes,
                           model_amplitudes,
                           model_amplitudes,
                           self.aa_grad_factor)
        b_part = np.einsum('j,kj',
                           model_amplitudes,
                           self.bb_grad_factor)
        d_amplitude = model_amplitudes - self.observed_amplitudes
        c_part = np.einsum("kj,j",
                           self.inv_cov_amplitudes,
                           d_amplitude)
        grad_log_posterior = -2*(a_part - b_part) - c_part
        #print("The gradient of the log-posterior is: " + str(grad_log_posterior))
        return grad_log_posterior