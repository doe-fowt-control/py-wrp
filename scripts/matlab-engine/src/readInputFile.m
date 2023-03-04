function [gravity, density_water, degrees_of_freedom, ...
    incident_wave_angle, draft_full, beam_full, length_full, ...
    operating_draft_full, scale, density_scale, displacement_full, ...
    waterplane_area_full, drag_coeff_heave, drag_coeff_roll, ...
    inertia_roll_full, C44_full, C34_full, ...
    spectral_truncation_parameter, peakedness_coefficient]...
    = readInputFile(file_name)
%% Open file
file_id = fopen(file_name,'rt');

%% Basic parameters
% gravity (m/s^2)
gravity     = readInputFileLine(file_id);
% Water density (kg/m^3)
density_water   = readInputFileLine(file_id);
% Nb. of dofs: 2 or 3
degrees_of_freedom  = readInputFileLine(file_id);

%% Irregular wave spectrum parameters
% Minimum significant wave amplitude for SPECTRUM = As_min (m)
minimum_spectral_amplitude  = readInputFileLine(file_id);
% Maximum significant wave amplitude for SPECTRUM = As_max (m)
maximum_spectral_amplitude  = readInputFileLine(file_id);
% Nb. of significant wave amplitude for SPECTRUM  NAs
number_spectral_amplitudes  = readInputFileLine(file_id);
% Minimum Peak spectral period (s) TP_min
minimum_peak_spectal_period = readInputFileLine(file_id);
% Maximum Peak spectral period (s) TP_max
maximum_peak_spectral_period = readInputFileLine(file_id);
% Nb. of Peak spectral period  NTP
number_peak_spectral_periods = readInputFileLine(file_id);
% Wave incident angle (deg)
incident_wave_angle = readInputFileLine(file_id);
% Fetch (m) (-1 for PM <> 0 for JS spectrum)
spectral_fetch_input = readInputFileLine(file_id);
% Wind speed at 10 m
spectral_wind_speed_input = readInputFileLine(file_id);
% Number of frequencies in each spectrum representation (e.g., 80)
number_spectral_frequencies = readInputFileLine(file_id);
% Parameter for spectral truncation at low/high frequency (e.g.: 0.0001)
spectral_truncation_parameter = readInputFileLine(file_id);
% Peakedness coefficient of JS spectrum
peakedness_coefficient = readInputFileLine(file_id);
% Simulation duration in number of peak period (old NTP)
number_periods_to_run = readInputFileLine(file_id);
% Max time for simulations (if non-zero use TCMAX rather that NTP)
time_maximum_input = readInputFileLine(file_id);
% Time step for reinterpolaiton of all results (no interp if 0)
time_step_for_interpolation = readInputFileLine(file_id);

%% Full scale data of the tug/float
% Float draft (m)
draft_full     = readInputFileLine(file_id);
% Float width (m)
beam_full     = readInputFileLine(file_id);
% Float length (m)
length_full     = readInputFileLine(file_id);
% Float operating draft (m)
operating_draft_full = readInputFileLine(file_id);
% Geometric Scale with respect to Prony coefficients
scale  = readInputFileLine(file_id);
% density Scale with respect to Prony coefficients
density_scale = readInputFileLine(file_id);
% Height of center of gravity above keel (m)
KGF    = readInputFileLine(file_id);
% Height of center of buoyancy above keel (m^2)
BGF    = readInputFileLine(file_id);
% Displacement (m^3)
displacement_full    = readInputFileLine(file_id);
% Waterplane area (m^2)
waterplane_area_full    = readInputFileLine(file_id);
% Distance between center of buoyancy and metacenter in pitch (m)
BMPF   = readInputFileLine(file_id);
% Distance between center of buoyancy and metacenter in roll (m)
BMRF   = readInputFileLine(file_id);

%% Mechanical properties and Linear hydrostatic stiffness matrix terms
% Float drag coefficients for heave motion
drag_coeff_heave    = readInputFileLine(file_id);
% Float drag coefficients for pitch motion
drag_coeff_pitch    = readInputFileLine(file_id);
% Inertia for pitch motion (kg.m^2)
inertia_pitch_full   = readInputFileLine(file_id);
% Float drag coefficients for roll motion
drag_coeff_roll   = readInputFileLine(file_id);
% Inertia for roll motion (kg.m^2)
inertia_roll_full  = readInputFileLine(file_id);
% Pitch restoring coefficient (N.m/rad)
C55_full = readInputFileLine(file_id);
% Heave-Pitch restoring coefficient (N/rad)
C35_full = readInputFileLine(file_id);
% Roll restoring coefficient (N.m/rad)
C44_full = readInputFileLine(file_id);
% Heave-Roll restoring coefficient (N/rad)
C34_full = readInputFileLine(file_id);
% Roll-pitch restoring coefficient (N.m/rad)
C45_full = readInputFileLine(file_id);

%% Local function definition
    function file_parameter = readInputFileLine(file_id)
        scanned_cell_array = textscan(file_id, '%f', 1, 'CommentStyle', '%');
        file_parameter = scanned_cell_array{1};
    end

end
