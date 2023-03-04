function [scale_inverse_sqrt, scale_squared, scale_third_power, ...
    scale_fourth_power, C_44, C_34, beam, operating_draft, ...
    equivalent_box_length, total_mass_heave, viscous_drag_factor_heave, ...
    total_inertia_roll, viscous_drag_factor_roll, ...
    AQWA_frequencies_scaled, scaled_displacement] ...
    = calculateScaledParameters(scale, density_scale, C44_full, ...
    C34_full, inertia_roll_full, draft_full, beam_full, length_full, ...
    displacement_full, waterplane_area_full, operating_draft_full, ...
    density_water, added_mass_infinity_heave, drag_coeff_heave, ...
    added_mass_infinity_roll, drag_coeff_roll, AQWA_frequencies)

% Scale derviatives for radiation force scaling
scale_inverse_sqrt = 1 / sqrt(scale);
scale_squared = scale ^ 2;
scale_third_power = scale ^ 3;
scale_fourth_power = scale ^ 4;

% Scaled hydrostatic coefficients
C_44 = C44_full * density_scale * scale_fourth_power;
C_34 = C34_full * density_scale * scale_third_power;

% Scaled geometry parameters
operating_draft = operating_draft_full * scale;
draft = draft_full * scale; 
beam = beam_full * scale;
length = length_full * scale;
S0 = waterplane_area_full * scale_squared;
SH = 2 * draft * (beam + length);
SR = length * draft; % Friction area for roll
equivalent_box_length = S0 / beam; % Equivalent box length
scaled_displacement = displacement_full * scale_third_power;

% Scaled heave mass
scaled_mass = density_water * density_scale * scaled_displacement; 
scaled_added_mass_heave = added_mass_infinity_heave * density_scale * scale_third_power; 
total_mass_heave = scaled_mass + scaled_added_mass_heave; % [kg]

% Scaled roll inertia
scaled_moment_of_inertia = inertia_roll_full * density_scale * scale_fourth_power * scale;
scaled_added_inertia_roll = added_mass_infinity_roll * density_scale * scale_third_power * scale_squared;
total_inertia_roll = scaled_moment_of_inertia + scaled_added_inertia_roll; % [kg*m^2]

% Scaled viscous factors
viscous_drag_factor_heave = 0.5 * density_water * density_scale * drag_coeff_heave * SH; % Float viscous drag factor for heave (kg/m)
viscous_drag_factor_roll = 0.5 * density_water * density_scale * drag_coeff_roll * SR * beam; % Float viscous drag factor for roll -- includes B/2 arm length + 2 surfaces  (kg/m)

% Scaled frequency for excitation force/moment 
AQWA_frequencies_scaled = AQWA_frequencies * scale_inverse_sqrt; 

end
