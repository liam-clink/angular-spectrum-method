"""
Run this to see the demonstration
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq

WIDTH = 5.0e-6
GRID_N = 2**10
WAVELENGTH = 1.0e-6  # m
SPATIAL_FREQUENCY = 1.0 / WAVELENGTH  # rad / m
print(SPATIAL_FREQUENCY)

x = np.linspace(-25.0e-6, 25.0e-6, GRID_N)
y = np.linspace(0.0, 50.0e-6, GRID_N)

X, Y = np.meshgrid(x, y, indexing="ij")


initial_intensity = np.zeros_like(x)
initial_intensity[(x > (-WIDTH / 2)) & (x < (WIDTH / 2))] = 1.0
initial_amplitude = initial_intensity**0.5

f_initial_amplitude = fft(initial_amplitude)
frequencies = fftfreq(GRID_N, d=x[1] - x[0])
print(frequencies)
#### Propagator
# for k_x <= k, propagator is np.exp(-1.j*dz*np.sqrt(k^2 - k_x^2))
# for k_x > k, propagator is np.exp(-dz*np.sqrt(k_x^2 - k^2))
# These need to be separated because np.sqrt(-1) isn't unique (branching)
# basically this is saying that the propagator is np.exp(-1.j*dz*k_z) plus enforcing the monochromatic condition
dy = y[1] - y[0]
propagator = np.ones_like(f_initial_amplitude)
# TODO: Do I need a 2pi factor in frequency space?
mask_propagating = np.abs(frequencies) <= SPATIAL_FREQUENCY
propagator[mask_propagating] = (
    np.exp(-1.0j * 2.0 * np.pi * dy * np.sqrt(SPATIAL_FREQUENCY**2 - frequencies**2))
)[mask_propagating]
mask_evanescent = np.abs(frequencies) > SPATIAL_FREQUENCY
propagator[np.abs(frequencies) > SPATIAL_FREQUENCY] = (
    np.exp(-2.0 * np.pi * dy * np.sqrt(-(SPATIAL_FREQUENCY**2) + frequencies**2))
)[mask_evanescent]

result = np.zeros_like(X)
f_old_amplitude = f_initial_amplitude
for j, _ in enumerate(y):
    f_new_amplitude = f_old_amplitude * propagator
    result[:, j] = ifft(f_new_amplitude)
    f_old_amplitude = f_new_amplitude

plt.pcolormesh(X, Y, np.abs(result) ** 2)
plt.gca().set_aspect("equal")
plt.show()
