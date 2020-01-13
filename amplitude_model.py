import numpy as np
import pyccl as ccl

class amplitude_model:
    """
    Class representing an amplitude (e.g., number of galaxies N) model as a function of redshift z
    """
    def __init__(self, redshift_highres_axis, redshift_lowres_axis, ells,
                 cosmo = ccl.Cosmology(Omega_b=0.05,Omega_c=0.25,sigma8=0.8,h=0.67,n_s=0.96,Omega_k=0)):
        # Initialize user input
        self.redshift_highres_axis = redshift_highres_axis
        self.redshift_lowres_axis = redshift_lowres_axis
        self.ells = ells
        self.cosmo = cosmo

        # Compute midpoints and edges
        self.redshift_low_edges = self.redshift_lowres_axis[:-1]
        self.redshift_high_edges = self.redshift_lowres_axis[1:]
        self.redshift_midpoints = 0.5 * (self.redshift_low_edges + self.redshift_high_edges)

        # Compute amplitudes with respect to redshift
        self.highres_amplitudes = self.get_highres_amplitudes()
        self.power_spectrum_slices = self.get_power_spectra()


    def get_amplitude(self, redshift):
        """
        Amplitude model N(z), here: a Gaussian
        """
        sigma = 0.2
        mean = 1.0
        return np.exp(-0.5 * ((redshift - mean) / sigma) ** 2) / np.sqrt(2 * np.pi * sigma ** 2)

    def evaluate_basis_function(self, redshift, redshift_lower_edge, redshift_higher_edge):
        """
        Basis function N(z) can be written as a sum of, here: heaviside
        """
        return (np.heaviside(redshift - redshift_lower_edge, 0.) - np.heaviside(redshift - redshift_higher_edge, 0.))\
               / (redshift_higher_edge - redshift_lower_edge)

    def get_highres_amplitudes(self):
        """
        Redshift distribution for each slice
        """
        highres_amplitudes = np.array([self.evaluate_basis_function(self.redshift_highres_axis, z_l, z_h)
                              for z_l, z_h in zip(self.redshift_low_edges, self.redshift_high_edges)])
        return highres_amplitudes

    def get_power_spectrum(self, cosmo, nz1, nz2):
        """
        Compute power spectrum between two redshift distributions
        """
        t1 = ccl.WeakLensingTracer(cosmo, nz1)
        t2 = ccl.WeakLensingTracer(cosmo, nz2)
        return ccl.angular_cl(cosmo, t1, t2, self.ells)

    def get_power_spectra(self):
        """
        Power spectra for each pair of slices
        """
        lowres_amplitude_dim = len(self.redshift_lowres_axis)-1
        power_spectrum_dim = len(self.ells)
        cls_slices = np.zeros([lowres_amplitude_dim, lowres_amplitude_dim, power_spectrum_dim])
        for i1 in range(lowres_amplitude_dim):
            for i2 in range(i1, lowres_amplitude_dim):
                cls_slices[i1, i2, :] = self.get_power_spectrum(self.cosmo,
                                                                (self.redshift_highres_axis,
                                                                 self.highres_amplitudes[i1]),
                                                                (self.redshift_highres_axis,
                                                                 self.highres_amplitudes[i2]))
                if i1 != i2:
                    cls_slices[i2, i1, :] = cls_slices[i1, i2, :]
        return cls_slices
