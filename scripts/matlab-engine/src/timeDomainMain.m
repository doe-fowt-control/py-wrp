clc;
close all;
clear;

%% Wave data !!need to get this from file!!

% load /Users/Shawn/Documents/+digital-twin/data/2.28.23/per_Hs03_Tp1_H01.4_Nw5_sense.mat
load /Users/Shawn/Documents/+digital-twin/data/2.28.23/irr_Hs03_Tp1_H01.4_Nw100_sense.mat

%% for figure saving
fname = '/Users/Shawn/Documents/figures/experiment/time-domain-v0/case8';

%% timing
fs = 100;
dt = 1/fs;

t_start = 15;    % simulation start time
t_end = 45;     % simulation end time
t_duration = t_end - t_start;

t_reinit = 0.05; % new simulation every _s
t_intermediate_duration = t_reinit - dt;

it_start = t_start * fs + 1;    % start index for sampling
it_end = t_end * fs;

% time array (start from zero
t_end_sim = t_duration - dt;
time = 0 : dt : t_end_sim;




%% sort data

% grab data from array: wave elevation, roll, heave pots. Means come from
% early sample (1s - 2s, after sensors stabilize). Multiply by appropriae scaling
% factors
eta_raw = all_data(it_start:it_end, 8)*0.06064506793;
eta_mean = mean(all_data(100:200, 8)*0.06064506793);
roll_raw = all_data(it_start:it_end, 10).*0.156;
roll_mean = mean(all_data(100:200, 10).*0.156);
heave_raw = all_data(it_start:it_end, 9).*0.156;
heave_mean = mean(all_data(100:200, 9)).*0.156;

% center by the expected mean
eta_centered = eta_raw - eta_mean;
roll_centered = roll_raw - roll_mean;
heave_centered = heave_raw - heave_mean;

% compute roll angle
dH = roll_centered - heave_centered;
roll = -asind(dH / 0.167);

% % final processed signals
% eta_measured = eta_centered;
% roll = roll;
% heave = heave_centered;

eta_measured = movmean(eta_centered, [0 0]);
roll = movmean(roll, [4 0]);
heave = heave_centered;

% signal rates
dheavedt = diff(heave) / dt;
drolldt = diff(roll) / dt;

%% verify processed data

% decompose wave elevation at float
[plant_amplitudes,...
    plant_phases,...
    plant_frequencies] = ...
    fft_decomp(fs, eta_measured);

eta_reconstructed = sum(plant_amplitudes.*cos(plant_frequencies'.*time + plant_phases));

f=figure;
subplot(3,1,1)
hold on
% plot(time, eta_reconstructed)
plot(time + t_start, eta_measured, 'DisplayName', 'wave elevation')
xlabel('time (s)')
ylabel('\eta (m)')

subplot(3,1,2)
plot(time + t_start, roll, 'DisplayName', 'roll measured')
xlabel('time (s)')
ylabel('roll (deg)')

subplot(3,1,3)
plot(time + t_start, heave, 'DisplayName', 'heave measured')
xlabel('time (s)')
ylabel('heave (m)')

% close


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
    scaled_displacement] = calculateScaledParameters(scale, ...
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

[C_44, operating_draft, total_inertia_roll, ~, scaled_displacement] = ...
    updateExperimentalFloatParameters(initial_total_inertia_roll, ...
    beam_model, density_water, number_control_mass_layers, ...
    control_rack_mass, control_rack_length, single_control_mass);

control_on_flag = 0;
%%

iter = 0;
end_time = 0;

full_roll = [];
full_heave = [];

initial_values = zeros(20,1);

while end_time < t_end_sim

    current_start_time = iter*t_reinit;
    end_time = current_start_time + t_intermediate_duration;
    solution_requested_times = current_start_time: dt : end_time;


    % Choose state function handle based on degrees of freedom
    [solution_time, solution_x] = ...
        ode113(@(t, x) ...
        seakeepingContinuousTime(t, x, gravity, density_water, ...
        total_mass_heave, total_inertia_roll, beam_model, operating_draft, ...
        equivalent_box_length, C_44, C_34, viscous_drag_factor_heave, ...
        viscous_drag_factor_roll, number_prony_heave, beta_heave_real, ...
        s_heave_real, beta_heave_imag, s_heave_imag, number_prony_roll, ...
        beta_roll_real, s_roll_real, beta_roll_imag, s_roll_imag, ...
        I_Re_3_start, I_Re_3_end, ...
        I_Im_3_start, I_Im_3_end, ...
        I_Re_4_start, I_Re_4_end, ...
        I_Im_4_start, I_Im_4_end, ...
        control_on_flag, scale, ...
        scaled_displacement, control_rack_length, control_mass, 0,...
        float_excitation_data,...
        eta_measured, time, fs),...
        solution_requested_times,...
        initial_values);

    current_roll = rad2deg(solution_x(:, 3));
    current_heave = solution_x(:, 1);
    
    full_roll = [full_roll; current_roll];
    full_heave = [full_heave; current_heave];
    
    % grab new states
    initial_values = solution_x(end, :)';
    
%     % replace with measured ones
    [~, it] = min(abs(end_time - time));

    
    initial_values(1) = heave(it);
    initial_values(3) = deg2rad(roll(it));      
    
    % d/dt needs to go back one index due to the nature of 'diff'
    initial_values(2) = dheavedt(it-1);
    initial_values(4) = deg2rad(drolldt(it-1));
    

    iter = iter + 1;
end
%%

% subplot(2,1,1)
% plot(full_roll)
% 
% subplot(2,1,2)
% plot(full_heave)

% subplot(3,1,1)
% grid on
% legend()

subplot(3,1,2)
hold on
grid on
plot(time + t_start, full_roll)
% plot(solution_time, rad2deg(solution_x(:, 3)), 'DisplayName', 'roll simulated')
ylim([-12, 12])
legend()


subplot(3,1,3)
hold on
grid on
plot(time + t_start, full_heave)
% plot(solution_time, solution_x(:, 1), 'DisplayName', 'heave simulated')
ylim([-0.03, 0.03])
legend()

lines = findobj(gcf,'Type','Line');
for i = 1:numel(lines)
  lines(i).LineWidth = 1;
end

%%
f.Position = [200 200 800 600];
%%
savefig([fname '.fig'])
print([fname '.png'], '-dpng', '-r500')

close all