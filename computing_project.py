"""
Calculating Hubble's Constant

Created on Fri Nov 10 13:37:16 2023

@author: yusufmspahi
"""
# Importing packages

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%%
# Optional packages for graphs

import scienceplots
plt.style.use(['science','no-latex'])

#%%
# Importing the data
spectral_data_path = r'C:\Users\Yusuf\OneDrive - Imperial College London\Computing\Computing Project\Data\Hbeta_spectral_data.csv' #path to the data
distance_data_path = r'C:\Users\Yusuf\OneDrive - Imperial College London\Computing\Computing Project\Data\Distance_Mpc.txt' #path to the data

# Using pandas to read raw data
spectral_df = pd.read_csv(spectral_data_path, skiprows=3, header=None)
distance_df = pd.read_csv(distance_data_path, sep='\t', skiprows=2, header=None, names=['Observation', 'Distance (Mpc)', 'Valid Instrument Response'])

# Converting to numpy arrays
spectral_data = np.array(spectral_df)
distance_data = np.array(distance_df)

# Extracting frequency and spectral intensity by indexing the numpy array [start:end:step, column]
frequency_data = spectral_data[::2, :]
spectral_intensity_data = spectral_data[1::2, :]

# frequency_data is a 2D array inculding observation number array([[2.899900e+04, 5.614273e+14, 5.614773e+14, ..., 6.113272e+14, 6.113772e+14, 6.114273e+14], ...])   

# Sorting the 2D arrays w.r.t. observation number
sorted_indices_frequency = np.argsort(frequency_data[:, 0])
sorted_frequency_data = frequency_data[sorted_indices_frequency]

sorted_indices_spectral = np.argsort(spectral_intensity_data[:, 0])
sorted_spectral_intensity_data = spectral_intensity_data[sorted_indices_spectral]

sorted_indices_distance = np.argsort(distance_data[:, 0])
sorted_distance_data = distance_data[sorted_indices_distance]

#%%
# Defining a function to plot raw data
def plot_spectral_data(frequency, intensity, observation_id, plot_number):
    
    plt.figure(figsize=(12, 8), dpi=1000)
    
    plt.plot(frequency, intensity, label=f'Observation {observation_id:.0f}')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectral Intensity (a.u.)')
    plt.title(f'Plot {plot_number} of Spectral Intensity vs Frequency')
    
    plt.legend()
    plt.grid(True)
    plt.show()
   
# Creating a for loop to plot raw data using the function above

num_plots = 30 # the number of observations
for i in range(0, num_plots):
    
    frequency = sorted_frequency_data[i, 1:] # the ith row excuding the observation number
    intensity = sorted_spectral_intensity_data[i, 1:] # the ith row excuding the observation number
    observation_id = sorted_frequency_data[i][0] # the observation number from the ith row
    plot_number = i+1
    
    plot_spectral_data(frequency, intensity, observation_id, plot_number)
    
#%%
# Defining line + gaussian function for curve fitting
def line_gaussian(x, gradient, intercept, amplitude, mean, stddev):
    
    gaussian = amplitude*np.exp(-(x-mean)**2/(2*stddev**2))
    
    line = gradient*x + intercept
    
    return line + gaussian

# Defining function to determine initial guesses for parameters in each curve fit
def initial_guesses(frequency_data_single_row, spectral_data_single_row):
    
    # Using polyfit to obtain gradient and intercept
    x = np.array([frequency_data_single_row[0], frequency_data_single_row[-1]])
    y = np.array([spectral_data_single_row[0], spectral_data_single_row[-1]])
    gradient, intercept = np.polyfit(x, y, 1)
    
    # Subtracting line from gaussian to find amplitude and peak frequency
    gaussian_data = spectral_data_single_row - (gradient * frequency_data_single_row + intercept)
    peak_frequency = frequency_data_single_row[np.argmax(gaussian_data)] # np.argmax gives the index of the largest element
    amplitude = max(gaussian_data)
    
    return gradient, intercept, amplitude, peak_frequency

# Defining funciton to plot the fitted curve over the raw data
def plot_spectral_data_with_fit(frequency, intensity, observation_id, plot_number, fitted_curve_x, fitted_curve_y):
    
    plt.figure(figsize=(12, 8), dpi=1000)
    
    plt.plot(frequency, intensity, label=f'Observation {observation_id:.0f}') # raw data
    plt.plot(fitted_curve_x, fitted_curve_y, label='fitted curve', color='orange', lw=2) # fitted curve
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectral Intensity (a.u.)')
    plt.title(f'Plot {plot_number} of Spectral Intensity vs Frequency with Fitted Curve')
    
    plt.legend(loc='best', frameon=True, edgecolor='black', facecolor='white', fancybox=False, framealpha=1)
    plt.grid(True)
    plt.show()
   
    
# for loop creates appends values to each list
sorted_observed_frequencies = [] 
sorted_observed_frequencies_unc = []
excluded_observation_numbers = [] # checks if all non-valid instument responses were excluded
valid_distance_data = [] # new list of data excluding non-valid instument responses

num_plots = 30
for i in range(0, num_plots):
    
    if sorted_distance_data[i,:][2] == 0: # if the valid instrument response = 0
        excluded_observation_numbers.append(sorted_distance_data[i,:][0])
        continue # continues to the next observation
    
    frequency = sorted_frequency_data[i, 1:] # the ith row excuding the observation number
    intensity = sorted_spectral_intensity_data[i, 1:] # the ith row excuding the observation number
    observation_id = sorted_frequency_data[i][0] # the observation number from the ith row
    plot_number = i+1
    
    g1, g1, g3, g4 = initial_guesses(frequency, intensity)
    initial_parameters = [g1, g1, g3, g4, 4e12] # rough estimate of sigma
    
    popt, pcov = curve_fit(line_gaussian, frequency, intensity, p0 = initial_parameters) #applying curve fit
    uncertainty_all = np.sqrt(np.diag(pcov)) # takes the square root of the diagonal of the covariance matrix to determine uncertainties in each parameter
    
    sorted_observed_frequencies.append(popt[3]) # appends the mean (observed freq.) obtained from the curve fit to the list
    sorted_observed_frequencies_unc.append(uncertainty_all[3]) # appends the uncertainty associated with each mean (observed freq.) to the list
    valid_distance_data.append(sorted_distance_data[i,:][1]) #appends data excluding non-valid instument responses
    
    fitted_curve_x = np.linspace(min(frequency), max(frequency), 1000) # creates x values for fitted curve
    fitted_curve_y = line_gaussian(fitted_curve_x, *popt) # using the function and optimal parameters from curve fit to obtain y values for fitted curve
    
    plot_spectral_data_with_fit(frequency, intensity, observation_id, plot_number, fitted_curve_x, fitted_curve_y) # uses the fuction to create plots with the fitted curve
    
#%%
# Calculating the velocities using the redshift formula

# Constants
c = 2.9979e8 # Speed of light in metres per second
lambda_e = 486.1e-9 # Emitted wavelength for Hydrogen-beta line in metres

# Converting lists to numpy arrays
observed_frequencies = np.array(sorted_observed_frequencies)
observed_frequencies_unc = np.array(sorted_observed_frequencies_unc)
distances_mpc = np.array(valid_distance_data)

# Converting observed frequencies to observed wavelengths
lambda_o = c/observed_frequencies

# Obtaining velocities by using the rearranged redshift formula
velocities = c*(lambda_o**2 - lambda_e**2)/(lambda_o**2 + lambda_e**2)
velocities_kms = velocities/1000

#%%
# Calculating uncertainies in the velocity using the error propagation formula

d_df = -c/(observed_frequencies**2) # derivative of 'c/f' w.r.t. f

lambda_o_unc = np.sqrt(d_df**2 * observed_frequencies_unc**2)

d_dlambda = 4*c*lambda_e**2*lambda_o/(lambda_e**2 + lambda_o**2)**2 # derivative of the rearranged redshift formula w.r.t. lambda

velocities_unc = np.sqrt(d_dlambda**2 * lambda_o_unc**2)

velocities_kms_unc = velocities_unc/1000

#%%
# Curve fitting to determine hubble's constant

def line(x, m, c):
    return m*x + c

h_popt, h_pcov = curve_fit(line, distances_mpc, velocities_kms, sigma = velocities_kms_unc, p0 = [70, 0])

hubbles_constant = h_popt[0]

unc_pcov = np.sqrt(np.diag(h_pcov))

hubbles_constant_unc = unc_pcov[0]

#%%
# Plotting the final graph

fitted_hubble_x = np.linspace(min(distances_mpc), max(distances_mpc), 1000) # creates x values for fitted curve
fitted_hubble_y = line(fitted_hubble_x, *h_popt) # creates y values for fitted curve

plt.figure(figsize=(12, 8), dpi=1500)

plt.plot(distances_mpc, velocities_kms,'o', label='Collected data', ms=5, color='C0')
plt.plot(fitted_hubble_x, fitted_hubble_y, label='Fitted Curve', color='C3', lw=1)
plt.text(100, 18000, f'$H_0$ = {hubbles_constant:.2f} $\pm$ {hubbles_constant_unc:.2f} km/s/Mpc', bbox=dict(facecolor='white', edgecolor='black'), size = 13)

plt.xlabel('Distance (Mpc)', size=12)
plt.ylabel('Redshift Velocity (km/s)', size=12)
plt.title(r"Determining Hubble's Constant", size=15)

plt.legend(loc='best', frameon=True, edgecolor='black', facecolor='white', fancybox=False, framealpha=1)
plt.grid(True)
#plt.savefig(r'C:\Users\Yusuf\OneDrive - Imperial College London\Computing\Computing Project\To Submit\hubbles_constant_plot.png')
plt.show()