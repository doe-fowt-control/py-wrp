%% Seakeeping Model Predictive Controller Simulation
%   This script runs a seakeeping model predictive controller simulation.
% The closed loop simulations run with a nonlinear model predictive
% controller, an extended Kalman filter for state estimation, disturbance
% previewing and output measurement feedback. The "plant" is a numerical
% model of the lab-scale float that will be run for experimental validation
% in the URI wave tank.
%   The disturbances to the float are the heave excitation force,
% the roll excitation force, and the wave elevation. The disturbances are
% also previewed. The true wave elevation, predicted wave evelation, and
% predicted spectral wave components are provided in
% WavePrediction\PredictionOffline\renwtdata. These results were run
% separately using the Stephan Grilli's numerical wave tank code and Shawn
% Albertson's wave reconstruction alorithm. There are both regular and
% irregular waves results, which are looped through in the simulations. The
% true spectral wave components are generated in this script using the NWT
% results and the loadNWTData function.
%   The simulations are run for varying prediction horizons. When the
% prediction horizon is 0 there is no control action, and the float  is
% allowed to react with no control. When the prediction horizon is 1, the
% controller is acting with only current information (i.e. no previewing).
% Lastly, several cases are run to see the effect of running the simulation
% with varying prediction horizons allowing for disturbance previewing.
%   This version runs with the getCodeGenerationData and
% generated MEX functions for a faster implementation of the simulations. A
% new MEX function is generated every simulation loop since every loop a
% new nlobj controller object is initialized and set with a different
% prediction horizon. nlobj controller objects and the generated MEX
% functions are not saved in each loop.
%   Generating each MEX function takes about a minute. For using this code
% with the experimental setup one set of controller settings should be
% chosen as the inputs to initialize_controller() and the MEX function
% should be generated ahead of runtime by running initialize_controller()
% as part of the setup stage before the experiments are run. Code is
% refactored in this version to have two main MATLAB functions
% initialize_controller() and update_controller() that can be called
% outside of MATLAB using the Python to MATLAB engine API, in preparation
% for running on the experimental setup.
%   23 February 2023, Stephanie Steele
clc;
close all;
clear;

% Print extra details
verbose = false;

% Specify folder to save results to
new_result_subfolder = 'Updated_Control_Mass\';

%% Numerical Wave Tank (NWT) data sets loop
% List of wave data to run through the simulation
NWT_data = ["periodic_re", "spectral_re", "spectral", "periodic_Tp1", ...
    "spectral_Tp1", ];

% Choose NWT data folders and files
for data_loop = 1 : length(NWT_data)
    % Periodic files from renwtdata
    if strcmp(NWT_data(data_loop), "periodic_re")
        NWT_data_folder = 'WavePrediction\PredictionOffline\renwtdata';
        wave_data_subfolder = 'Periodic_renwt\';
        prediction_spectral_components_file = [NWT_data_folder...
            '\per_A_phi_omega.csv'];
        wave_elevation_file = [NWT_data_folder ...
            '\per_eta_true_predicted.csv'];
        % Irregular files from renwtdata
    elseif strcmp(NWT_data(data_loop), "spectral_re")
        NWT_data_folder = 'WavePrediction\PredictionOffline\renwtdata';
        wave_data_subfolder = 'Spectral_renwt\';
        prediction_spectral_components_file = [NWT_data_folder...
            '\irreg_A_phi_omega.csv'];
        wave_elevation_file = [NWT_data_folder ...
            '\irreg_eta_true_predicted.csv'];
        % Irregular files from Tp1 data
    elseif strcmp(NWT_data(data_loop), "spectral_Tp1")
        NWT_data_folder = 'WavePrediction\PredictionOffline\renwtdata';
        wave_data_subfolder = 'Spectral_Tp1\';
        prediction_spectral_components_file = [NWT_data_folder...
            '\Tp1.0_irreg_A_phi_omega.csv'];
        wave_elevation_file = [NWT_data_folder ...
            '\Tp1.0irreg_eta_true_predicted.csv'];
        % Periodic files from Tp1 data
    elseif strcmp(NWT_data(data_loop), "periodic_Tp1")
        NWT_data_folder = 'WavePrediction\PredictionOffline\renwtdata';
        wave_data_subfolder = 'Periodic_Tp1\';
        prediction_spectral_components_file = [NWT_data_folder...
            '\Tp1.0_periodic_A_phi_omega.csv'];
        wave_elevation_file = [NWT_data_folder ...
            '\Tp1.0_periodic_eta_true_predicted.csv'];
        % Irregular files from extra data set
    elseif strcmp(NWT_data(data_loop), "spectral")
        NWT_data_folder = 'WavePrediction\PredictionOffline\NWT-stephanie';
        wave_data_subfolder = 'Spectral\';
        prediction_spectral_components_file = [NWT_data_folder...
            '\A_phi_omega.csv'];
        wave_elevation_file = [NWT_data_folder '\eta_true_predicted.csv'];
    else
        return;
    end
    
    if ismac
        NWT_data_folder = strrep(NWT_data_folder, "\", "/");
        wave_data_subfolder = strrep(wave_data_subfolder, "\", "/");
        prediction_spectral_components_file = strrep(...
            prediction_spectral_components_file, "\", "/");
        wave_elevation_file = strrep(wave_elevation_file, "\", "/");
    end

    % Load respective NWT data and generate associated spectral data
    [time_array_NWT_data, NWT_wave_elevation, NWT_frequencies, ...
        NWT_amplitudes, NWT_phases, prediction_wave_elevation, ...
        prediction_frequencies, prediction_amplitudes, ...
        prediction_phases, peak_period] = loadNWTData(...
        prediction_spectral_components_file, wave_elevation_file);
    plant_angular_frequencies = NWT_frequencies; % 1 x m row vector
    plant_amplitudes = NWT_amplitudes; % 1 x m row vector
    plant_phases = NWT_phases; % 1 x m row vector

    %% Simulation parameters and NWT post-processing
    % Prediction horizons for control cases
    controller_sampling_time = 0.05; % [s]
    control_on_flag_array = [0, 1, 1, 1];
    one_step_in_cycles = controller_sampling_time / peak_period;
    pred_horizons_in_cycles = [one_step_in_cycles, ...
        one_step_in_cycles, 0.1, 0.25]; % #cycles
    pred_horizon_array = round(pred_horizons_in_cycles * ...
        peak_period / controller_sampling_time); % #steps

    % Float simulation parameters based on NWT data length
    NWT_end_time = time_array_NWT_data(end);
    longest_prediction_duration = max(pred_horizons_in_cycles) * ...
        peak_period;
    duration_time = NWT_end_time - longest_prediction_duration; % [s]
    time_array = 0 : controller_sampling_time : duration_time; %[s]
    time_array_full = 0 : controller_sampling_time : NWT_end_time;

    % Sample plant and predicted wave signals at controller sampling time
    NWT_wave_sampled = interp1(time_array_NWT_data, NWT_wave_elevation, ...
        time_array_full);
    plant_wave_signal = NWT_wave_sampled'; % n x 1 column vector
    predicted_wave_sampled = interp1(time_array_NWT_data, ...
        prediction_wave_elevation, time_array_full);
    predicted_wave_signal = predicted_wave_sampled';

    %% Control cases loop
    % Make results folder with subfolders
    results_folder = join(['Results/', new_result_subfolder, ...
        wave_data_subfolder]);
    mkdir(results_folder);

    % Open text file to save simulation durations to
    file_id = fopen(join([results_folder ...
        '/Control_Cases_Durations.txt']), 'w');
    fopen(file_id);

    % Initialize data matrices for plotting figures
    number_cases = length(control_on_flag_array);
    number_timesteps = length(time_array);
    roll_plant_cases = zeros(number_cases, number_timesteps);
    roll_predicted_cases = zeros(number_cases, number_timesteps);
    mv_cases = zeros(number_cases, number_timesteps);

    for control_case = 1 : length(control_on_flag_array)
        % Controller inputs
        control_on_flag = control_on_flag_array(control_case);
        prediction_horizon = pred_horizon_array(control_case);

        % Initialize controller (MEX function + EKF)
        [float_excitation_data, parameters_cell_array, onlineData, EKF,...
            y, x, controller_rack_length, n_x, ~, n_md, nlobj] = ...
            initializeController(prediction_horizon, control_on_flag, ...
            controller_sampling_time);

        % Initialize the closed loop simulation
        states_history = zeros(n_x, length(time_array));
        state_estimates_history = zeros(n_x, length(time_array));
        plant_md_history = zeros(n_md, length(time_array));
        predicted_md_history = zeros(n_md, length(time_array));
        mv_history = zeros(1, length(time_array));

        % Initialize controller rack, starts centrally
        mv = controller_rack_length / 2;
        u_EKF = [mv, 0, 0, 0];

        %% Closed-loop time domain simulations
        t_start = datetime('now','TimeZone','local','Format','HH:mm:ss');
        for timestep = 1 : length(time_array)
            % Update time
            time = time_array(timestep);

            %% Update measured disturbances predictions
            % Use most recent prediction for spectral components, a new
            % prediction every 1 second
            desired_prediction_time = floor(time); % [s]
            set_length = 100;
            prediction_indices = (1 : set_length) + ...
                set_length * desired_prediction_time;
            predicted_amplitude = prediction_amplitudes(...
                prediction_indices);
            predicted_ang_freq = prediction_frequencies(...
                prediction_indices);
            predicted_phase = prediction_phases(prediction_indices);

            % Preview wave elevations
            prediction_horizon_indices = timestep : 1 : ...
                timestep + (prediction_horizon - 1);
            predicted_preview_wave_elevation = ...
                predicted_wave_signal(prediction_horizon_indices);

            % Match prediction set timing -- the time for the prediction
            % sets go from 0 to 1s only, not continuously from 0 to 18.4s
            original_prediction_horizon_time = ...
                time_array_full(prediction_horizon_indices)';
            remainder_time = rem(original_prediction_horizon_time, 1);
            prediction_horizon_times = remainder_time;

            %% Update controller position
            % Calculate new controller position
            [mv, u_EKF, onlineData, xk, md_prediction] = ...
                updateController(EKF, y, u_EKF, parameters_cell_array, ...
                float_excitation_data, prediction_horizon_times, ...
                predicted_amplitude, predicted_ang_freq, ...
                predicted_phase, predicted_preview_wave_elevation, mv, ...
                onlineData);

            %% Update float simulation
            % Implement first optimal control move with true measured
            % distubances as inputs to plant model
            [total_excitation_heave, total_excitation_roll] ...
                = excitationForceCalculator(float_excitation_data, ...
                time, plant_amplitudes, plant_angular_frequencies, ...
                plant_phases);
            md_plant = [total_excitation_heave, total_excitation_roll, ...
                plant_wave_signal(timestep)];
            u_plant = [mv, md_plant];
            x = seakeepingDiscrete(x, u_plant, parameters_cell_array{:});

            % Generate sensor data
            y = measuredOutputFunction(x);

            %% Save variables for plotting and optional printouts
            % Save plant states and state estimates
            states_history(:, timestep) = x;
            state_estimates_history(:, timestep) = xk;

            % Save mv history
            mv_history(timestep) = mv;

            % Save plant and predicted measured disturbances
            plant_md_history(:, timestep) = md_plant';
            predicted_md_history(:, timestep) = md_prediction(1, :)';

            % Print extra details
            if verbose
                % Print progress to console
                formatSpec = '%4.2f %% complete\n';
                fprintf(formatSpec, timestep / length(time_array) * 100);

                % Display optimization solution details
                disp(onlineData);

                % Can use to debug numerical perterbation method
                % and analytic Jacobian differences / errors
                % There is a very small discrepancy at A(2,1) in the state
                % Jaconbian which I haven't been able to figure out
                validateFcns(nlobj, x, mv, md_plant, ...
                    parameters_cell_array);
            end
        end

        %% End time domain simulation, save simulation duration, make plots
        t_end = datetime('now','TimeZone','local','Format', 'HH:mm:ss');
        simulation_duration = t_end - t_start;
        fprintf(file_id, '%23s\n', strcat('Control Case', {' '}, ...
            num2str(control_case), {' '}, 'duration:', {' '}, ...
            string(simulation_duration)));

        plotSummaryEachControlCase(time_array, states_history, ...
            state_estimates_history, plant_md_history, ...
            predicted_md_history, mv_history, results_folder, ...
            pred_horizons_in_cycles, pred_horizon_array, control_case);

        roll_plant_cases(control_case, :) = states_history(3, :);
        roll_predicted_cases(control_case, :) = ...
            state_estimates_history(3, :);
        mv_cases(control_case, :) = mv_history;

        % End control cases loop
    end

    % Plot figure over all control cases for each NWT data
    plotPaperFigureAllCases(time_array, plant_md_history, ...
        predicted_md_history, roll_plant_cases, roll_predicted_cases, ...
        mv_cases, results_folder, pred_horizons_in_cycles, ...
        pred_horizon_array);

    % Close out timestamps text file
    fclose(file_id);

    % End NWT data loop
end