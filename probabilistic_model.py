from amplitude_model import *

class probabilistic_model:
    """
    Class representing an amplitude (e.g., number of galaxies N) model as a function of redshift z
    """
    def __init__(self, amplitude_model, observed_amplitudes, inv_covar_amplitudes, observed_power_spectra, inv_cov_power_spectra):
        # Initialize user input
        # Theory
        self.power_spectra = amplitude_model.power_spectra # cl_slices
        self.lowres_amplitudes = amplitude_model.lowres_amplitudes
        # Observations
        self.observed_amplitudes = observed_amplitudes
        self.inv_covar_amplitudes = inv_covar_amplitudes
        self.observed_power_spectra = observed_power_spectra
        self.inv_cov_power_spectra = inv_cov_power_spectra



    """ Computes the (-2*)log posterior, gaussian due to the conjugate prior
    Input: self.observed_power_spectra, cl_slices, covar_cl, nz, covar_nz
    Output: (-2*)log_posterior at given data
    """
    def get_log_posterior(self):
        ##### Log-likelihood #####
        cl_theory = np.einsum('i,ijk,j', self.observed_amplitudes, self.power_spectra, self.observed_amplitudes)
        res = self.observed_power_spectra - cl_theory
        log_likelihood = np.einsum('i,ij,j',
                                   res, self.inv_cov_power_spectra, res)
        print("The (-2*)log-likelihood i(s: " + str(log_likelihood))

        ##### Log-prior #####
        observed_amplitudes_mean = np.mean(self.observed_amplitudes)
        #sigma_nz = 0.1 * self.lowres_amplitudes + 0.02 * np.amax(self.lowres_amplitudes)
        #covar_nz = np.diag(sigma_nz ** 2)
        #inv_covar_nz = np.linalg.inv(covar_nz)
        log_prior = np.einsum('i,ij,j', (self.observed_amplitudes - observed_amplitudes_mean), self.inv_covar_amplitudes, (self.observed_amplitudes - observed_amplitudes_mean))
        print("The (-2*)log-prior is: " + str(log_prior))

        ##### Log-posterior #####
        log_posterior = log_likelihood + log_prior
        print("The (-2*)log-posterior is: " + str(log_posterior))
        return log_posterior

    def get_grad_log_posterior(self):
        grad_log_posterior = -2 * np.einsum('j,m,n,kja,mnb,ab', self.observed_amplitudes, self.observed_amplitudes, self.observed_amplitudes,
                                            self.power_spectra, self.power_spectra, self.inv_cov_power_spectra) \
                             + 2 * np.einsum('j,kja,ab,b', self.observed_amplitudes, self.power_spectra, self.inv_cov_power_spectra, self.observed_power_spectra) \
                             + np.einsum('kj,j', self.inv_covar_amplitudes, (self.observed_amplitudes - np.mean(self.observed_amplitudes)))
        return grad_log_posterior

if __name__ == '__main__':
    ### Use case example ###

    # User input: 3 axes
    z_lores = np.linspace(0, 2, 11)
    z_hires = np.linspace(0, 2.5, 1280)
    ells = np.linspace(100, 2000, 100)

    # Build model
    model = amplitude_model(z_hires, z_lores, ells)

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