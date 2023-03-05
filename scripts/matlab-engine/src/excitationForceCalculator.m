%% Calculate excitation forces in heave and roll 
% This function can be used with either estimated wave periodic or spectral
% amplitude, angular frequency, and phase decomposition to input into
% controller and extended Kalman filter models or a priori "true" wave
% spectral decomposition to input into plant model.
% Inputs:
% float_excitation_data stucture containing the following 5 fields:
% AQWA_frequencies_scaled, 
% excitation_amplitude_heave, 
% excitation_phase_heave,
% excitation_amplitude_roll,
% excitation_phase_roll: each are 1 x 100 data inputs from AQWA dataset
% time: can be scalar of length(time) x 1 column vector
% free_surface_amplitudes, angular_frequencies, free_surface_phases: each
% are a 1 x (number of spectral components) row vector of wave spectral
% decomposition
% Outputs:
% total_excitation_force_heave, total_excitation_force_roll: length(time) x
% 1 column vector of resulting excitation forces for input into models

function [total_excitation_force_heave, total_excitation_force_roll] ...
    = excitationForceCalculator(float_excitation_data, time, ...
    free_surface_amplitudes, angular_frequencies, free_surface_phases)

% Unpackage float excitation data structure
AQWA_frequencies_scaled = float_excitation_data.AQWA_frequencies_scaled;
excitation_amplitude_heave = float_excitation_data.excitation_amp_heave;
excitation_phase_heave = float_excitation_data.excitation_phase_heave; 
excitation_amplitude_roll = float_excitation_data.excitation_amp_roll;
excitation_phase_roll = float_excitation_data.excitation_phase_roll;

% Interpolate excitation force data, make sure extrapolation values are set
% to 0
excitation_amplitude_heave_interpolated = interp1(...
    AQWA_frequencies_scaled, excitation_amplitude_heave, ...
    angular_frequencies, 'spline', 0);
excitation_phase_heave_interpolated= interp1(...
    AQWA_frequencies_scaled, excitation_phase_heave, ...
    angular_frequencies, 'spline', 0);
excitation_amplitude_roll_interpolated = interp1(...
    AQWA_frequencies_scaled, excitation_amplitude_roll, ...
    angular_frequencies, 'spline', 0);
excitation_phase_roll_interpolated = interp1(...
    AQWA_frequencies_scaled, excitation_phase_roll, ...
    angular_frequencies, 'spline', 0);


% Total excitation forces
total_excitation_force_heave = sum(excitation_amplitude_heave_interpolated .* free_surface_amplitudes .* cos(angular_frequencies .* time + excitation_phase_heave_interpolated + free_surface_phases), 2);
total_excitation_force_roll = sum(excitation_amplitude_roll_interpolated .* free_surface_amplitudes .* cos(angular_frequencies .* time + excitation_phase_roll_interpolated + free_surface_phases), 2);
end