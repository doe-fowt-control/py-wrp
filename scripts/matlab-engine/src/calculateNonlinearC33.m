function [C_33, C1, C3, geometry_level_1, geometry_level_2] = ...
    calculateNonlinearC33(relative_heave, gravity, ...
    density_water, operating_draft, equivalent_box_length, beam_model, ...
    scale, scaled_displacement)

% This function should return a scalar C33 for one time step, and an array
% for multiple time steps
%   Returns the non-linear heave d restoring coefficient (C33) for the
% warping tug due to geometric differences at different heave amplitudes 
% in Newtons/meter.
%   z = instantaneous heave position with respect to wave position eta, 
% in meters
%   Function assumes a constant beam of 7.3025 meters, the operating draft 
% is 0.66 meters, and no coupling with pitch or roll at full scale, values 
% are scaled here. operating_draft is already at scale.
%   Correction to scaling factors added 28 Sept 2022, S. Steele

% Scaled coefficients
C0 = scaled_displacement; % scaled submerged volume
C1 = 5.893; % non-dimensional
C2 = 22.807 * scale; % scaled C2 coefficient 
C3 = 2.979; % non-dimensional
C4 = 25.360 * scale; % scaled C4 coefficient

% Scaled levels on the hull where geometry changes
geometry_level_1 = 0.759 * scale;
geometry_level_2 = 1.517 * scale;
geometry_level_3 = 2.4257 * scale;

C_33 = zeros(length(relative_heave), 1);
Lv = C4; % Initialize Lv for MEX code generation's sake
for i = 1 : length(relative_heave)
    if relative_heave(i) < -operating_draft
        C_33(i) = gravity * density_water * C0; % rare airborne case
    else
        if (-operating_draft < relative_heave(i)) && (relative_heave(i) <= geometry_level_1 - operating_draft)
            Lv = equivalent_box_length + C1 * relative_heave(i);      % For z=0, Lv=LV0 and Lv*B = S0
        elseif (geometry_level_1 - operating_draft < relative_heave(i)) && (relative_heave(i) <= geometry_level_2 - operating_draft)
            Lv = C2 + C3 * relative_heave(i);
        elseif (geometry_level_2 - operating_draft < relative_heave(i)) && (relative_heave(i) <= geometry_level_3 - operating_draft)
            Lv = C4;
        elseif relative_heave(i) > geometry_level_3 - operating_draft
            Lv = C4;
        end
        C_33(i) = gravity * density_water * beam_model * Lv;
    end
end






























