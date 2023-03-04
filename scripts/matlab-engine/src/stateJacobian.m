function [A, Bmv] = stateJacobian(x, u, gravity, density_water, ...
    total_mass_heave, total_inertia_roll, beam_model, operating_draft, ...
    equivalent_box_length, C_44, C_34, viscous_drag_factor_heave, ...
    viscous_drag_factor_roll, ~, beta_heave_real, s_heave_real, ...
    beta_heave_imag, s_heave_imag, ~, beta_roll_real, s_roll_real, ...
    beta_roll_imag, s_roll_imag, ~, ~, ~, ~, ~, ~, ~, ~, ...
    control_on_flag, scale, scaled_displacement, ~, control_mass, ~)
%% State function Jacobian with respect to states x vector
eta = u(4); 
relative_heave = x(1) - eta;
[C_33, C1, C3, geometry_level_1, geometry_level_2]  = ...
    calculateNonlinearC33(relative_heave,  gravity, density_water, ...
    operating_draft, equivalent_box_length, beam_model, scale, ...
    scaled_displacement);  

if (-operating_draft < relative_heave) && (relative_heave <= geometry_level_1 - operating_draft)
    partialC33_partialx1 = density_water * gravity * beam_model * C1;
elseif (geometry_level_1 - operating_draft < relative_heave) && (relative_heave <= geometry_level_2 - operating_draft)
    partialC33_partialx1 = density_water * gravity * beam_model * C3;
else
    partialC33_partialx1 = 0;
end

A = zeros(length(x), length(x));

A(1, 2) = 1;
A(2, 1) =  (-partialC33_partialx1 * x(1) - C_33) / total_mass_heave;
A(2, 2) = -2 * viscous_drag_factor_heave * abs(x(2)) / total_mass_heave;
A(2, 3) = -C_34 * cos(x(3)) / total_mass_heave;
A(2, 5) = -beta_heave_real(1) / total_mass_heave;
A(2, 6) = -beta_heave_real(2) / total_mass_heave;
A(2, 7) = -beta_heave_real(3) / total_mass_heave;
A(2, 8) = -beta_heave_real(4) / total_mass_heave;
A(2, 9) = beta_heave_imag(1) / total_mass_heave;
A(2, 10) = beta_heave_imag(2) / total_mass_heave;
A(2, 11) = beta_heave_imag(3) / total_mass_heave;
A(2, 12) = beta_heave_imag(4) / total_mass_heave;

% dz3 / dx
A(3, 4) = 1;

% dz4 / dx
A(4, 1) = -C_34 / total_inertia_roll;
A(4, 3) = -C_44 * cos(x(3)) / total_inertia_roll;
A(4, 4) = -0.5 * beam_model^2 * viscous_drag_factor_roll * abs(x(4)) / total_inertia_roll;
A(4, 13) = -beta_roll_real(1) / total_inertia_roll;
A(4, 14) = -beta_roll_real(2) / total_inertia_roll;
A(4, 15) = -beta_roll_real(3) / total_inertia_roll;
A(4, 16) = -beta_roll_real(4) / total_inertia_roll;
A(4, 17) = beta_roll_imag(1) / total_inertia_roll;
A(4, 18) = beta_roll_imag(2) / total_inertia_roll;
A(4, 19) = beta_roll_imag(3) / total_inertia_roll;
A(4, 20) = beta_roll_imag(4) / total_inertia_roll;

% dz5-8 / dx
A(5, 2) = 1;
A(5, 5) = s_heave_real(1);
A(5, 9) = -s_heave_imag(1);
A(6, 2) = 1;
A(6, 6) = s_heave_real(2);
A(6, 10) = -s_heave_imag(2);
A(7, 2) = 1;
A(7, 7) = s_heave_real(3);
A(7, 11) = -s_heave_imag(3);
A(8, 2) = 1;
A(8, 8) = s_heave_real(4);
A(8, 12) = -s_heave_imag(4);

% dz9-12 / dx
A(9, 5) = s_heave_imag(1);
A(9, 9) = s_heave_real(1);
A(10, 6) = s_heave_imag(2);
A(10, 10) = s_heave_real(2);
A(11, 7) = s_heave_imag(3);
A(11, 11) = s_heave_real(3);
A(12, 8) = s_heave_imag(4);
A(12, 12) = s_heave_real(4);

% dz13-16 / dx
A(13, 4) = 1;
A(13, 13) = s_roll_real(1);
A(13, 17) = -s_roll_imag(1);
A(14, 4) = 1;
A(14, 14) = s_roll_real(2);
A(14, 18) = -s_roll_imag(2);
A(15, 4) = 1;
A(15, 15) = s_roll_real(3);
A(15, 19) = -s_roll_imag(3);
A(16, 4) = 1;
A(16, 16) = s_roll_real(4);
A(16, 20) = -s_roll_imag(4);

% dz17-20 / dx
A(17, 13) = s_roll_imag(1);
A(17, 17) = s_roll_real(1);
A(18, 14) = s_roll_imag(2);
A(18, 18) = s_roll_real(2);
A(19, 15) = s_roll_imag(3);
A(19, 19) = s_roll_real(3);
A(20, 16) = s_roll_imag(4);
A(20, 20) = s_roll_real(4);

%% State function Jacobian with respect to manipulated variable u_mv
Bmv = zeros(length(x), 1);
if control_on_flag 
    Bmv(4, 1) = (2 * control_mass * gravity) / total_inertia_roll;
end

end