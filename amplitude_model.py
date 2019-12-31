import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl

class amplitude_model:
    """
    Class representing an amplitude (e.g., number of galaxies N) model as a function of redshift z
    """
    def __init__(self, redshift_highres_axis, redshift_lowres_axis, power_spectrum_axis,
                 cosmo = ccl.Cosmology(Omega_b=0.05,Omega_c=0.25,sigma8=0.8,h=0.67,n_s=0.96,Omega_k=0)):
        # Initialize user input
        self.redshift_highres_axis = redshift_highres_axis
        self.redshift_lowres_axis = redshift_lowres_axis
        self.power_spectrum_axis = power_spectrum_axis
        self.cosmo = cosmo

        # Compute midpoints and edges
        self.redshift_low_edges = self.redshift_lowres_axis[:-1]
        self.redshift_high_edges = self.redshift_lowres_axis[1:]
        self.redshift_midpoints = 0.5 * (self.redshift_low_edges + self.redshift_high_edges)

        # Compute amplitudes with respect to redshift
        self.highres_amplitudes = self.get_highres_amplitudes()
        self.lowres_amplitudes = self.get_lowres_amplitudes()
        self.amplitude_steps = self.get_amplitude_steps()

        # Compute power spectra
        self.power_spectrum_steps = self.get_power_spectrum_steps()
        self.power_spectra = self.get_power_spectra()

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
        # Redshift distribution for each slice
        highres_amplitudes = np.array([self.evaluate_basis_function(self.redshift_highres_axis, z_l, z_h)
                              for z_l, z_h in zip(self.redshift_low_edges, self.redshift_high_edges)])
        return highres_amplitudes

    def get_lowres_amplitudes(self):
        # In each slice we take the value of the Gaussian above at the centre of the bin
        lowres_amplitudes = self.get_amplitude(self.redshift_midpoints)
        lowres_amplitudes /= np.sum(lowres_amplitudes)
        return lowres_amplitudes

    def get_amplitude_steps(self):
        return np.einsum('j,ji', self.lowres_amplitudes, self.highres_amplitudes)

    def get_power_spectrum(self, cosmo, nz1, nz2):
        # Compute power spectrum between two redshift distributions
        t1 = ccl.WeakLensingTracer(cosmo, nz1)
        t2 = ccl.WeakLensingTracer(cosmo, nz2)
        return ccl.angular_cl(cosmo, t1, t2, self.power_spectrum_axis)

    def get_power_spectrum_steps(self):
        # Power spectrum for the model redshift distribution
        cls_steps = self.get_power_spectrum(self.cosmo,
                           (self.redshift_highres_axis, self.amplitude_steps),
                           (self.redshift_highres_axis, self.amplitude_steps))
        return cls_steps

    def get_power_spectra(self):
        # Power spectra for each pair of slices
        lowres_amplitude_dim = len(self.lowres_amplitudes)
        power_spectrum_dim = len(self.power_spectrum_axis)
        cls_slices = np.zeros([lowres_amplitude_dim, lowres_amplitude_dim, power_spectrum_dim])
        for i1 in range(lowres_amplitude_dim):
            for i2 in range(i1, lowres_amplitude_dim):
                cls_slices[i1, i2, :] = self.get_power_spectrum(self.cosmo,
                                               (self.redshift_highres_axis, self.highres_amplitudes[i1]),
                                               (self.redshift_highres_axis, self.highres_amplitudes[i2]))
                if i1 != i2:
                    cls_slices[i2, i1, :] = cls_slices[i1, i2, :]

        return cls_slices

    def get_power_spectra_sandwich(self, power_spectra):
        # Now sandwich with N(z) amplitudes
        cls_sandwich = np.einsum('i,ijk,j', self.lowres_amplitudes, power_spectra, self.lowres_amplitudes)
        return cls_sandwich

    def plot_amplitudes(self):
        # Compare high-resolution N(z) and low-resolution slicing
        nz_smooth = self.get_amplitude(self.redshift_highres_axis)
        plt.figure()
        for n in self.highres_amplitudes:
            plt.plot(self.redshift_highres_axis, n, 'k-', lw=1)
        plt.plot(self.redshift_highres_axis, nz_smooth, 'r-')
        plt.plot(self.redshift_highres_axis, self.amplitude_steps, 'b-')
        plt.show()

    def plot_power_spectra(self):
        plt.figure()
        plt.plot(self.power_spectrum_axis, self.power_spectrum_steps, 'b-')
        plt.plot(self.power_spectrum_axis, self.get_power_spectra_sandwich(self.power_spectra), 'r--')
        plt.loglog()
        plt.show()

