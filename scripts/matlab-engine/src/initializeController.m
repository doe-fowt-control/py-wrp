function [float_excitation_data, parameters_cell_array, onlineData, EKF,...
    y, x, control_rack_length, n_x, n_y, n_md, nlobj] = ...
    initializeController(prediction_horizon, control_on_flag, ...
    controller_sampling_time)
% This function initializes the nonlinear model predictive controller, with
% an internal model of the float and an external extended Kalman filter. 
% The controller is compiled into the MEX function 'mexController' which is
% generated when you run this function. The extended Kalman filter is used
% for state estimation for the controller and needs to be passed in
% 'update_controller' as an argument.
% There are 3 inputs to this function:
%   prediction_horizon is the prediction horizon you want the nonlinear 
% model predictive controller to have, in number of steps, or control 
% intervals. For example, a prediction horizon of 5 steps for a controller 
% sampling time of 0.01 s will have the controller predict 0.05 s into the 
% future.
%   control_on_flag should be set to 0 for no control and 1 for control.
% For use with the experimental setup via Python, this should always be 1.
%   controller_sampling_time is the sampling time for the controller, in
% seconds.
%   
% The first 4 outputs of this function are related to the controller
% initialization. Output the first 4 outputs to pass to the function 
% 'update_controller' when you call this function from Python to conduct 
% the wave tank float experiments. These outputs are:
%   float_excitation_data is a structure containing the scaled and
% interpolated AQWA excitation force response per unit amplitude data for
% the lab-scale float. Pass this structure to update_controller() to
% help calculate total excitation forces on the float.
%   parameters_cell_array is a cell array of the float model parameters
% used in nonlinear model predictive controller's internal models. Pass
% this cell array directly to update_controller().
%   onlineData is the optimized data structure object that the MEX compiled
% controller function expects as an input. Pass this to update_controller()
% and update_controller() will update and pass onlineData back out as an
% output with the updated current measured disturbance predictions as as of
% its fields. 
%   EKF is the extended Kalman filter object used in conjunction with the
% nonlinear model predictive controller. Pass this object to
% update_controller(), where it will be updated automatically (and modified 
% without being passed back out of update_controller() as an output).
%   The last 7 are outputs related to float simulation initialization, 
% plotting, and debugging details in the MATLAB simulation script main.m,
% but you can use y as an output to initialize the experimental measured
% outputs (heave and roll angle measurements) if you like.

%% Float Model Parameters
% Reformat AQWA excitation force and added mass at infinity data
[excitation_amplitudes, excitation_phases, AQWA_frequencies, ...
    AQWA_incident_wave_angles, number_AQWA_frequencies, ...
    added_mass_infinity] = reformatAQWAData();
added_mass_infinity_heave  = added_mass_infinity(3);
added_mass_infinity_roll  = added_mass_infinity(4);

% Read user input file for simulation parameters
input_file_name  = 'Inputs_Periodic_Waves_Heave_Roll_Lab_Scale.txt';
[gravity, density_water, degrees_of_freedom, incident_wave_angle, ...
    draft_full, beam_full, length_full, operating_draft_full, ...
    scale, density_scale, displacement_full, waterplane_area_full, ...
    drag_coeff_heave, drag_coeff_roll, inertia_roll_full, C44_full, ...
    C34_full, ~, ~]...
    = readInputFile(input_file_name);

% Calculate scaled hydrostatic and mechanical properties
[scale_inverse_sqrt, scale_squared, scale_third_power, ...
    scale_fourth_power, initial_C_44, C_34, beam_model, ...
    initial_operating_draft, equivalent_box_length, total_mass_heave, ...
    viscous_drag_factor_heave, initial_total_inertia_roll, ...
    viscous_drag_factor_roll, AQWA_frequencies_scaled, ...
    initial_scaled_displacement] = calculateScaledParameters(scale, ...
    density_scale, C44_full, C34_full, inertia_roll_full, draft_full, ...
    beam_full, length_full, displacement_full, waterplane_area_full, ...
    operating_draft_full, density_water, added_mass_infinity_heave, ...
    drag_coeff_heave, added_mass_infinity_roll, drag_coeff_roll, ...
    AQWA_frequencies);

% Extract, scale, and interpolate excitation (Froude-Krylov) force
[excitation_amplitude_heave, excitation_phase_heave, ...
    excitation_amplitude_roll, excitation_phase_roll] ...
    = interpolateExcitationAndResponse(number_AQWA_frequencies, ...
    AQWA_incident_wave_angles, excitation_amplitudes, ...
    excitation_phases, incident_wave_angle, density_scale, ...
    scale_squared, scale_third_power);

% Package scaled and interpolated spectral AQWA excitation force response 
% data (per unit amplitude) for calculating total excitation forces on
% float
float_excitation_data.AQWA_frequencies_scaled = AQWA_frequencies_scaled;
float_excitation_data.excitation_amp_heave = excitation_amplitude_heave;
float_excitation_data.excitation_phase_heave = excitation_phase_heave;
float_excitation_data.excitation_amp_roll = excitation_amplitude_roll;
float_excitation_data.excitation_phase_roll = excitation_phase_roll;

% Read Prony coefficients from files
file_name_heave = 'Prony_Coefficients_Heave.txt';
file_name_roll = 'Prony_Coefficients_Roll.txt';
[number_prony_heave, number_prony_roll, beta_heave_real, ...
    beta_heave_imag, s_heave_real, s_heave_imag, beta_roll_real, ...
    beta_roll_imag, s_roll_real, s_roll_imag] ...
    = readPronyCoefficientsRollHeave(file_name_heave, file_name_roll, ...
    density_scale, scale_inverse_sqrt, scale_squared, scale_fourth_power);

% Index numbers for unknown complex Prony functions "I"
number_states_heave_roll = 2 * degrees_of_freedom;
I_Re_3_start = number_states_heave_roll + 1;
I_Re_3_end = number_states_heave_roll + number_prony_heave;
I_Im_3_start = I_Re_3_end + 1;
I_Im_3_end = I_Re_3_end + number_prony_heave;
I_Re_4_start = I_Im_3_end + 1;
I_Re_4_end = I_Im_3_end + number_prony_roll;
I_Im_4_start = I_Re_4_end + 1;
I_Im_4_end = I_Re_4_end + number_prony_roll;

%% Controller and Simulation Settings
control_rack_mass = 0.687; % as measured,[kg]
control_rack_length = 0.635; % as measured, [m]
single_control_mass = 0.633; % as measured, [kg]
number_control_mass_layers = 1;
control_mass = number_control_mass_layers * single_control_mass; % 

[C_44, operating_draft, total_inertia_roll, inertial_moment_arm, ...
    scaled_displacement] = ...
    updateExperimentalFloatParameters(initial_total_inertia_roll, ...
    beam_model, density_water, number_control_mass_layers, ...
    control_rack_mass, control_rack_length, single_control_mass);

% % Check updated float parameters at command line
% initial_C_44
% C_44
% initial_operating_draft
% operating_draft
% initial_total_inertia_roll
% total_inertia_roll
% rough_measurement_inertial_moment_arm = 0.1269 % [m]
% inertial_moment_arm
% initial_scaled_displacement
% scaled_displacement

n_x = 20; % states x: 4 motion + 16 unknown complex Prony functions "I"
n_y = 2; % outputs y: 1 heave + 1 roll string potentiometer measurements
n_md = 3; % measured disturbance inputs: 1 excitation force in heave,
% 1 excitation force in roll, 1 wave elevation

%% Initialize closed loop time domain simulation
% Initialize new nlmpc object
nlobj = nlmpc(n_x, n_y, 'MV', 1, 'MD', [2 3 4]);
nlobj.Ts = controller_sampling_time;
nlobj.PredictionHorizon = prediction_horizon;
nlobj.ControlHorizon = prediction_horizon;
nlobj.Model.StateFcn = 'seakeepingContinuous';
nlobj.Jacobian.StateFcn = 'stateJacobian';
nlobj.Model.OutputFcn = 'measuredOutputFunction';
nlobj.Jacobian.OutputFcn = 'outputJacobian';
nlobj.Weights.OutputVariables = [0 1];
nlobj.Weights.ManipulatedVariablesRate = 0.02;
nlobj.MV.Min = 0;
nlobj.MV.Max = control_rack_length;

parameters_cell_array = {gravity, density_water, ...
    total_mass_heave, total_inertia_roll, ...
    beam_model, operating_draft, equivalent_box_length, ...
    C_44, C_34, ...
    viscous_drag_factor_heave, viscous_drag_factor_roll, ...
    number_prony_heave, beta_heave_real, s_heave_real, beta_heave_imag, s_heave_imag, ...
    number_prony_roll,  beta_roll_real, s_roll_real, beta_roll_imag, s_roll_imag, ...
    I_Re_3_start, I_Re_3_end, I_Im_3_start, I_Im_3_end, ...
    I_Re_4_start, I_Re_4_end, I_Im_4_start, I_Im_4_end, ...
    control_on_flag, ...
    scale, scaled_displacement, ...
    control_rack_length, control_mass, controller_sampling_time};

nlobj.Model.NumberOfParameters = length(parameters_cell_array);

% Validate user provided functions with MPC toolbox check
x0 = zeros(n_x, 1);
mv0 = control_rack_length / 2;
md0 = [0, 0, 0.005];
validateFcns(nlobj, x0, mv0, md0, parameters_cell_array);

% Generate optimal data structures for use with nlmpcmoveCodeGeneration
[coreData,onlineData] = getCodeGenerationData(nlobj, x0, mv0, ...
    parameters_cell_array);
onlineData.ref = [0 0];

% Initialize extended Kalman filter and measured outputs for unmeasured 
% state estimation
EKF = extendedKalmanFilter(@seakeepingDiscrete, ...
    @measuredOutputFunction);
x = zeros(n_x, 1);
y = zeros(n_y, 1);
EKF.State = x;

coder.extrinsic('optimoptions')

% Generate MEX function to be used in main.m
% buildMEX(nlobj, 'mexController', coreData, onlineData);
end