function [spectrum_wave, spectral_frequency_array, delta_frequency] = ...
    calculateJONSWAPSpectrum(maximum_frequency, minimum_frequency, ...
    number_spectral_frequencies, peakedness_coefficient, ...
    spectral_amplitude, spectral_peak_frequency, gravity)

% This function calculates the JONSWAP wave spectrum.
% Inputs:
% maximum_frequency: positive scalar, puts a maximum cap on the generated 
%   JONSWAP wave spectrum.
% minimum_frequency: positive scalar, puts a minimum cap on the generated
%   JONSWAP wave spectrum.
% number_spectral_frequencies: positive scalar, number of wave frequency
%   components in the generated spectrum.
% peakedness_coefficient: positive scalar, peakedness parameter of JONSWAP 
%   spectrum.
% spectral_amplitude: positive scalar, significant wave amplitude 
%   specification for JONSWAP spectrum.
% spectral_peak_frequency: positive scalar, peak frequency of JONSWAP
%   spectrum.
% gravity: positive scalar, gravitational constant.
% Output:
% spectrum_wave: 1 x number_spectral_frequencies vector, containing
%   JONSWAP spectral values at each spectral component frequency.

% Calculate base spectrum
delta_frequency = (maximum_frequency - minimum_frequency) / ...
    (number_spectral_frequencies - 1);
spectral_frequency_array = minimum_frequency + delta_frequency * ...
    (0 : 1 : number_spectral_frequencies - 1);
nondimensional_spectral_frequency_array = spectral_frequency_array /...
    spectral_peak_frequency;
spectrum_initial = exp(-1.25 ./ nondimensional_spectral_frequency_array ...
    .^ 4) ./ nondimensional_spectral_frequency_array .^ 5;

% Modify according to small spectral values
sigma(1 : number_spectral_frequencies) = 0.09;
indices_small_frequency_values = nondimensional_spectral_frequency_array ...
    <= 1;
sigma(indices_small_frequency_values) = 0.07;
spectrum_wave_initial = spectrum_initial .* peakedness_coefficient .^ ...
    (exp(-(nondimensional_spectral_frequency_array - 1) .^ 2 ./ (2 .* ...
    sigma .^ 2)));

% Modify according to alpha and lambda parameters
lambda = trapz(nondimensional_spectral_frequency_array, ...
    spectrum_wave_initial); 
alpha  = spectral_amplitude ^ 2 * (spectral_peak_frequency ^ 4) / ...
    (4 * lambda * gravity ^ 2);
spectrum_wave = spectrum_wave_initial * alpha * gravity ^ 2 / ...
    spectral_peak_frequency ^ 5;
end
