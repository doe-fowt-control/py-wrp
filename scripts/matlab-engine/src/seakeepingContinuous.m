function dxdt = seakeepingContinuous(x, u, gravity, density_water, ...
    total_mass_heave, total_inertia_roll, beam_model, operating_draft, ...
    equivalent_box_length, C_44, C_34, viscous_drag_factor_heave, ...
    viscous_drag_factor_roll, number_prony_heave, beta_heave_real, ...
    s_heave_real, beta_heave_imag, s_heave_imag, number_prony_roll, ...
    beta_roll_real, s_roll_real, beta_roll_imag, s_roll_imag, ...
    I_real_heave_start, I_real_heave_end, I_imag_heave_start, ...
    I_imag_heave_end, I_real_roll_start, I_real_roll_end, ...
    I_imag_roll_start, I_imag_roll_end, control_on_flag, scale, ...
    scaled_displacement, control_rack_length, control_mass, ~)

% Measured disturbance inputs
mv = u(1);
total_excitation_force_heave = u(2);
total_excitation_force_roll = u(3);
eta = u(4);

% Restoring forces
relative_heave = x(1) - eta;
C_33  = calculateNonlinearC33(relative_heave,  gravity, density_water, ...
    operating_draft, equivalent_box_length, beam_model, scale, ...
    scaled_displacement);  
restoring_force_heave = C_33 * x(1);   
restoring_force_roll = C_44 * sin(x(3));
restoring_force_heave_roll = C_34 * sin(x(3));
restoring_force_roll_heave = C_34 * x(1);

% Viscous friction forces (without NL surface correction)
velocity_heave = x(2);       
viscous_force_heave = viscous_drag_factor_heave * velocity_heave * abs(velocity_heave);  %[N]
velocity_roll   = 0.5 * beam_model * x(4);  %(L/2)*d alpha_R/dt
viscous_force_roll = viscous_drag_factor_roll * velocity_roll * abs(velocity_roll);  %[Nm]

% Radiative damping forces (must be real): real(sum(Beta*I)) = real(sum(beta_real+i*beta_imag)*(I_real+i*I_imag))) = beta_real*I_real - beta_imag*I_imag
radiative_damping_memory_force_heave = sum(beta_heave_real(1 : number_prony_heave) .* x(I_real_heave_start : I_real_heave_end) - beta_heave_imag(1 : number_prony_heave) .* x(I_imag_heave_start : I_imag_heave_end));
radiative_damping_memory_force_roll = sum(beta_roll_real(1 : number_prony_roll) .* x(I_real_roll_start : I_real_roll_end) - beta_roll_imag(1 : number_prony_roll) .* x(I_imag_roll_start : I_imag_roll_end));

% Roll control action force
control_rack_position = mv; % 0 when control rack at the leftmost position
if control_on_flag
    control_action_roll = (2 * control_rack_position - control_rack_length) * control_mass * gravity;
else
    % Cases with no control action
    control_action_roll = 0;
end

% Initialize dxdt_ arrays for MEX function codegen's sake
dxdt_Re_I_heave = zeros(number_prony_heave, 1);
dxdt_Im_I_heave = zeros(number_prony_heave, 1);
dxdt_Re_I_roll = zeros(number_prony_roll, 1);
dxdt_Im_I_roll = zeros(number_prony_roll, 1);

% ODE's for state variables (includes scaling of radiative force)
dxdt1 = x(2);
dxdt2 = (total_excitation_force_heave - viscous_force_heave - radiative_damping_memory_force_heave - restoring_force_heave  - restoring_force_heave_roll) / total_mass_heave;   % Heave accel.
dxdt3 = x(4);
dxdt4 = (total_excitation_force_roll - viscous_force_roll - radiative_damping_memory_force_roll - restoring_force_roll - restoring_force_roll_heave + control_action_roll) / total_inertia_roll;  % Roll accel.
dxdt_Re_I_heave(1:number_prony_heave,1) = x(2) + s_heave_real(1 : number_prony_heave) .* x(I_real_heave_start : I_real_heave_end) - s_heave_imag(1 : number_prony_heave) .* x(I_imag_heave_start : I_imag_heave_end);
dxdt_Im_I_heave(1:number_prony_heave,1) = s_heave_imag(1 : number_prony_heave) .* x(I_real_heave_start : I_real_heave_end) + s_heave_real(1 : number_prony_heave) .* x(I_imag_heave_start : I_imag_heave_end);
dxdt_Re_I_roll(1:number_prony_roll,1) = x(4) + s_roll_real(1 : number_prony_roll) .* x(I_real_roll_start : I_real_roll_end) - s_roll_imag(1 : number_prony_roll) .* x(I_imag_roll_start : I_imag_roll_end);
dxdt_Im_I_roll(1:number_prony_roll,1) =  s_roll_imag(1 : number_prony_roll) .* x(I_real_roll_start : I_real_roll_end) + s_roll_real(1 : number_prony_roll) .* x(I_imag_roll_start : I_imag_roll_end);

dxdt  = [dxdt1; dxdt2; dxdt3; dxdt4; dxdt_Re_I_heave; dxdt_Im_I_heave; dxdt_Re_I_roll; dxdt_Im_I_roll];