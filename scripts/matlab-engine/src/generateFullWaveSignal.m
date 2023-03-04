function [full_wave_signal, spectral_frequencies, ...
    free_surface_amplitude_n, free_surface_phase_n] ...
    = generateFullWaveSignal(periodic_true, amplitude, ...
    angular_frequency, time, peakedness_coefficient, ...
    spectral_truncation_parameter, gravity, max_AQWA_frequencies_scaled,...
    min_AQWA_frequencies_scaled)

% This function generates the complete wave signal, in sampling time
% increment steps, for the entirety of the simulation. The signal is used
% to give current and preview wave elevations to the nonlinear model
% predictive controller (nlmpc) object, as measured disturbances. The
% generated signal can be either a regular, periodic wave or an irregular,
% spectral wave based on the Donelan et al (1985) formulation of the
% JONSWAP spectrum.
% Inputs:
% periodic_true: boolean, where 1 indicates periodic wave generation and 0
%   indicates spectral wave generation.
% amplitude: scalar, specifies either the amplitude for periodic waves or
%   the significant wave amplitude (Hs / 2) for spectral waves.
% angular_frequency: scalar, specifices either the angular frequency for
%   periodic waves or the peak spectral frequency for spectral waves.
% time: 1 x (number of timesteps + excitation force prediction horizon) 
%   row vector, the vector in time steps of time for the entire simulation.
% The below inputs are optional parameters that need to be specified for
% spectral wave generation only:
% peakedness_coefficient: scalar, steepness parameter commonly denoted as γ
%   (gamma) in JONSWAP spectrum equations.
% spectral_truncation_parameter: scalar, spectral truncation parameter
%   relative from peak as fraction of spectral peak for low and high
%   frequency truncation.
% gravity: scalar, gravity used in wave generation.
% max_AQWA_frequencies_scaled, min_AQWA_frequencies_scaled: each are a
%   scalar value, maximum and minimum frequencies from the AQWA dataset,
%   scaled for comparison to model conditions, used to make sure that
%   spectral generation does not exceed available AQWA data needed to
%   calculate total excitation forces.
% Outputs:
% full_wave_signal: length(time) x 1 column vector, as the function 
%   nlmpcmove expects a column vector for each specified measured 
%   disturbance, wave elevations in timesteps for the full duration of the 
%   simulation.
% free_surface_amplitude_n, free_surface_phase_n: each is a 1 x 
%   (number of spectral frequencies) row vector, spectral amplitude and
%   spectral random phase components of the total wave elevation.
% spectral_frequencies: 1 x (number of spectral frequencies) row vector, 
%   consisting of the frequencies contained in the spectrum.
clc;
close all;

% Select mode for generating periodic waves
if periodic_true == 1
    full_wave_signal = amplitude * cos(angular_frequency * time');
    free_surface_amplitude_n = amplitude;
    free_surface_phase_n = 0;
    spectral_frequencies = angular_frequency;
    
% Select mode for generating spectral waves
else
    % Specify general JONSWAP spectrum parameters
    spectral_peak_frequency = angular_frequency;
    spectral_amplitude = amplitude;
    number_spectral_frequencies  = 500;

    % Generate JONSWAP spectrum for initial low and high frequency cutoffs
    low_frequency  = 0.1;        % Low Frequency cut off (20s, 624 m DW)
    high_frequency  = 39.95;     % High frequency cutoff (.1s, 1.56 cm DW)
    [spectrum_wave, spectral_frequencies] = ...
        calculateJONSWAPSpectrum(high_frequency, low_frequency, ...
        number_spectral_frequencies, peakedness_coefficient, ...
        spectral_amplitude, spectral_peak_frequency, gravity);

    % Determine new high and low frequency cutoffs based on trunction
    % parameter and initial spectral values and make sure spectrum does not
    % exceed existing AQWA data cover.
    indices_small_spectrum_values = find(spectrum_wave >= ...
        max(spectrum_wave) * spectral_truncation_parameter);
    min_frequency_spectrum = max([min(spectral_frequencies(...
        indices_small_spectrum_values)) min_AQWA_frequencies_scaled]);
    max_frequency_spectrum = min([max(spectral_frequencies(...
        indices_small_spectrum_values)) max_AQWA_frequencies_scaled]);

    % Recalculate spectrum based on new cutoff frequencies
    [spectrum_wave_final, spectral_frequencies_final, ...
        delta_frequency_final] = calculateJONSWAPSpectrum(...
        max_frequency_spectrum, min_frequency_spectrum, ...
        number_spectral_frequencies, peakedness_coefficient, ...
        spectral_amplitude, spectral_peak_frequency, gravity);

    % Generate free surface representation by random phase method based on
    % spectrum
    free_surface_amplitude_n = sqrt(2 .* spectrum_wave_final * ...
        delta_frequency_final);
    free_surface_phase_n = 2 * pi * rand(1, number_spectral_frequencies);

    % Generate final full spectrum irregular waves signal
    eta    = sum(free_surface_amplitude_n' .* cos(...
        spectral_frequencies_final' * time + free_surface_phase_n'), 1);
    full_wave_signal = eta';

    % Plot generated JONSWAP spectrum
    figure;
    title('Ocean Wave Spectrum', 'fontsize', 16);
    xlabel('f (Hz)', 'fontsize', 16);
    ylabel('S (m^2. s)', 'fontsize', 16);
    set(gca, 'FontSize', 16);
    grid on;
    plot(spectral_frequencies_final / (2 * pi), spectrum_wave_final);
    xlabel('Frequency Components [rad/s]');
    ylabel('Spectrum [m^2 / rad/s')
end

% Plot generated full wave signal
figure;
plot(time, full_wave_signal);
xlabel('Time [s]');
ylabel('Wave Signal [m]');
end