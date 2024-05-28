
# Measuring Hubble’s Constant

### Introduction

In the field of cosmology, which studies the origin and evolution of the universe, one of the most fundamental observations is that of Hubble’s Law. The universe is expanding, which means the distance between objects is increasing with time. Hubble’s Law states that the velocity at which objects, such as galaxies, are moving away from the Earth is proportional to their distance from Earth.

$$\Huge{ v = H_0 D }$$

where $v$ is the velocity, $D$ is the distance, and $H_0$ is the value of Hubble’s constant.

The distance to a galaxy is inferred from the brightness of the galaxy. The velocity is inferred from the Doppler-shift of light emitted by the galaxy. For a galaxy that is moving away from the observer, the light will appear to the observer to be “redshifted” to longer wavelengths. The relationship between the velocity of the galaxy and the shift in wavelengths is given by the redshift formula:

$$\Huge \frac{\lambda_o}{\lambda_e} = \sqrt{\frac{1 + \frac{v}{c}}{1 - \frac{v}{c}}}$$

Where $\lambda_o$ and $\lambda_e$ are the observed and emitted wavelengths of light, $c = 2.9979 \times 10^8 \ \text{m/s}$ is the speed of light, and $v$ is the velocity of the emitter. The Hydrogen-beta spectral line (Hβ) is a deep-red visible line with a wavelength of 486.1 nm. This line is commonly observed in the emission spectra from galaxies.

Figure 1 illustrates how $H_0$ can be inferred from measurements of distance and the “redshifted” velocity.


<div align="center">
<img src='https://github.com/yusufmspahi/HubblesConstant/assets/170480436/1907d7a8-0802-45b6-bc7e-44cd8320071a' />
</div>

Fig. 1: A plot of velocity inferred from redshift versus distance for a range of galaxies. The slope of the line of best fit gives an estimate for $H_0$.

### Project Objective

Given 30 observations of the redshifted Hβ line from different galaxies and the distance to each galaxy, write a code in Python to calculate a value for Hubble’s constant. The output from your code should be a plot of velocity inferred from redshift vs. distance for each galaxy, the value of Hubble’s constant inferred from fitting this plot and its uncertainty.

### Data

Here is a description of the data that you are given:

- **Hbeta_spectral_data.csv**: Contains data for the observed shift of the Hβ spectral line. This file consists of 3 header rows, below which are 60 rows of data in pairs, with each pair corresponding to a different observation. The first row of a pair is frequency in Hz, and the second row is the spectral intensity in arbitrary units. The observation number is in the first column of each row.
- **Distance_Mpc.txt**: A file in which the first two rows are a header. It has three columns: observation number, the measured distance to that galaxy in Mpc, and the instrument response. Note that the order of the observation numbers in the “distances_Mpc” and “Hbeta_spectral_data” files are different.

<div align="center">
<img src='https://github.com/yusufmspahi/HubblesConstant/assets/170480436/9661e685-20d5-4bee-bb42-bcbccef08c52' />
</div>

Fig. 2: A plot of spectral intensity vs. frequency. The blue line shows the observed data. The orange line is obtained by fitting a straight line + Gaussian function to the data.

### Project Notes

- The data for spectral intensity versus frequency is noisy. To calculate the velocity, the data should be fitted with a combination of a straight line and a Gaussian function. The mean value of the fitted Gaussian is a good estimate for the observed frequency, from which an observed wavelength, $\lambda_o$, of the shifted Hβ line can be calculated.
- The instrument response for each observation is either good (value of 1) or bad (value of 0). Observations with bad instrument response should not be used in the calculation of $H_0$.
- Matching the observation number in the spectral data file with the observation number in the distance data file is necessary since the order of observation numbers is different in both files.

