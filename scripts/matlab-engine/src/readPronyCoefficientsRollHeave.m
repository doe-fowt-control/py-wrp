function [number_prony_coefficients_heave, ...
    number_prony_coefficients_roll, beta_heave_real, ...
    beta_heave_imaginary, s_heave_real, s_heave_imaginary, ...
    beta_roll_real, beta_roll_imaginary, s_roll_real, s_roll_imaginary] ...
    ...
    = readPronyCoefficientsRollHeave(file_name_heave, file_name_roll, ...
    density_scale, scale_inverse_sqrt, scale_squared, scale_fourth_power)

% Load heave Prony coefficents
prony_heave_data = load(file_name_heave);
[length_prony_file_heave, ~] = size(prony_heave_data);
number_prony_coefficients_heave = length_prony_file_heave / 2;
beta_heave_real = prony_heave_data(1 : number_prony_coefficients_heave, 1) * density_scale * scale_squared; % kg/s^2 or N/m
beta_heave_imaginary = prony_heave_data(1 : number_prony_coefficients_heave, 2) * density_scale * scale_squared; % kg/s^2
s_heave_real = prony_heave_data(number_prony_coefficients_heave + 1 : 2 * number_prony_coefficients_heave, 1) * scale_inverse_sqrt; % s^-1
s_heave_imaginary = prony_heave_data(number_prony_coefficients_heave + 1 : 2 * number_prony_coefficients_heave, 2) * scale_inverse_sqrt; % s^-1

%Load roll Prony coefficients
prony_roll_data = load(file_name_roll);
[length_prony_file_roll, ~] = size(prony_roll_data);
number_prony_coefficients_roll = length_prony_file_roll / 2;
beta_roll_real = prony_roll_data(1 : number_prony_coefficients_roll, 1) * density_scale * scale_fourth_power; % N.m
beta_roll_imaginary = prony_roll_data(1 : number_prony_coefficients_roll, 2) * density_scale * scale_fourth_power; % N.m
s_roll_real = prony_roll_data(number_prony_coefficients_roll + 1 : 2 * number_prony_coefficients_roll, 1) * scale_inverse_sqrt; % s^-1
s_roll_imaginary = prony_roll_data(number_prony_coefficients_roll + 1 : 2 * number_prony_coefficients_roll, 2) * scale_inverse_sqrt; % s^-1

end