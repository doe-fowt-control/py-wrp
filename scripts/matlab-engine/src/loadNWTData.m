function [time_array_NWT_data, NWT_wave_elevation, NWT_frequencies, ...
    NWT_amplitudes, NWT_phases, prediction_wave_elevation, ...
    prediction_frequencies, prediction_amplitudes, prediction_phases, ...
    peak_period] = loadNWTData(prediction_spectral_components_file, ...
    wave_elevations_file)

% Load wave and spectral components data
prediction_spectral_components = load(prediction_spectral_components_file);
wave_elevations = load(wave_elevations_file);

% Separate the wave elevation data
NWT_wave_elevation = wave_elevations(1, :);
prediction_wave_elevation = wave_elevations(2, :);

% Separate spectral components data
prediction_amplitudes_padded = prediction_spectral_components(1, :);
prediction_phases_padded = prediction_spectral_components(2, :);
prediction_frequencies_padded = prediction_spectral_components(3, :);

% The spectrum data all are padded with zeros; strip the first second.
sampling_rate_NWT_data = 100; % Hz
index_1s = sampling_rate_NWT_data + 1;
prediction_amplitudes = prediction_amplitudes_padded(index_1s : end);
prediction_phases = prediction_phases_padded(index_1s : end);
prediction_frequencies = prediction_frequencies_padded(index_1s : end);

% Make the time data
timestep_NWT_data = 1 / sampling_rate_NWT_data;
index_array_NWT_data = (0 : length(NWT_wave_elevation) - 1);
time_array_NWT_data = timestep_NWT_data * index_array_NWT_data;

% Make the NWT spectral components data
if ispc
    addpath WavePrediction\PredictionOffline\NWT-stephanie;
elseif ismac
    addpath WavePrediction/PredictionOffline/NWT-stephanie;
end

[NWT_amplitudes, NWT_phases, NWT_frequencies] = ...
    fft_decomp(sampling_rate_NWT_data, NWT_wave_elevation);

% Find NWT peak period 
[~, NWT_ang_freq_index] = max(NWT_amplitudes);
NWT_ang_freq_peak = NWT_frequencies(NWT_ang_freq_index);
peak_period = 2 * pi / NWT_ang_freq_peak; % [s]

end