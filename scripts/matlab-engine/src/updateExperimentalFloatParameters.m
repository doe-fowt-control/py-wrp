function [C_44, draft, updated_inertia_roll, inertial_moment_arm,...
    scaled_displacement] = ...
    updateExperimentalFloatParameters(initial_inertia_roll, beam_model, ...
    density_water, number_control_mass_layers, control_rack_mass, ...
    control_rack_length, single_control_mass)

% total_mass_heave = 0;
% scaled_displacement = 0;

%% Compute mass of full system. 
% Constants measured experimentally in-person
mass_motor_base = 1.02; % [kg] unweighted roll mechanism mass 
% (tube + motor + rollers)
mass_motor_assembled = mass_motor_base + 2 * number_control_mass_layers ...
    * single_control_mass; % [kg] 
% loaded roll mechanism mass

mass_barge_dry = 24.95 / 2.205; % [kg] unweighted barge
mass_heave_staff = 0.729; % [kg]
mass_roll_mech = 1.103; % [kg]
mass_barge_full = mass_barge_dry + mass_motor_assembled ...
    + mass_heave_staff + mass_roll_mech; % assembled barge mass

%% Geometry of the barge in terms of sloped surfaces
slope_aft = 13.3/4.1;
slope_fore = 22.3/8;    
flat_width = 0.9488;

%% Calculate draft
% based on displaced mass and geometry
fun = @(draft) (flat_width * draft + 1 / 2 * draft ^ 2 * (slope_aft ...
    + slope_fore)) * beam_model * density_water - mass_barge_full;
d = fzero(fun, 0.04);

if d > 0.041
    % case where you need to account for step change in geometry
    m1 = (flat_width * 0.041 + 1 / 2 * 0.041 ^ 2 * (slope_aft ...
        + slope_fore)) * density_water * beam_model;
    dm = mass_barge_full - m1;
    A2 = @(diff_draft) (1.1488 * diff_draft + 1 / 2 * diff_draft ^ 2 ...
        * (slope_fore)) - dm / beam_model / density_water;
    diff_draft = fzero(A2, 0.0005);
    draft = 0.041 + diff_draft;
else
    % otherwise, use what you got
    draft = d;
end

%% Center of buoyancy
% relative to keel, based on volume displaced by calculated draft
if draft < 0.041
    z = draft;
    numerator = (1 / 2 * 0.9488 * draft ^ 2) + (1 / 3 * (slope_aft ...
        + slope_fore) * draft ^ 3);
    denominator = 0.9488 * z + (1 / 2 * (slope_aft + slope_fore) ...
        * draft ^ 2);
    V = denominator * beam_model; % volume displaced water (m^3)
    cob = numerator / denominator;
elseif draft > 0.041
    h1 = 0.041;
    num1 = (1 / 2 * 0.9488 * h1 ^ 2) + (1 / 3 * (slope_aft ...
        + slope_fore) * h1 ^ 3);
    num2 = (1 / 3 * slope_fore * (draft ^ 3 - h1 ^ 3));
    d1 = 0.9488 * h1 + (1 / 2 * (slope_aft + slope_fore) * h1 ^ 2);
    d2 = (1/2 * slope_fore * (draft ^ 2 - h1 ^ 2));
    V = (d1 + d2) * beam_model; % volume displaced water (m^3)
    cob = (num1 + num2) / (d1 + d2);
end

scaled_displacement = V;
KB = cob;

%% BM
if draft < 0.041
    effective_length = flat_width + (slope_aft + slope_fore) * draft;
elseif draft > 0.041
    effective_length = 1.208 + slope_fore * draft;
end

I = 1 / 12 * effective_length * beam_model ^ 3; % waterplane moment of inertia
BM = I / V;

%% COG - moving ballast system 
% with respect to keel
deck_height = .1325; % includes mounting platform on which mechanism sits. otherwise 12.6
if number_control_mass_layers == 1
    cog_ballast = 0.0171 + deck_height;
    cog_mechanism = 0.02859 + deck_height;
elseif number_control_mass_layers == 2
    cog_ballast = 0.03283 + deck_height;
    cog_mechanism = 0.0272 + deck_height;
end

%% COG - Barge
% based on old cog plus moving ballast mechanism. doesn't include the
% heave staff or roll structure, as it doesn't contribute to GM
cog_dry = 0.071; % [m] as measured out of water 'tipping' experiment
KG = (cog_mechanism * mass_motor_assembled + cog_dry * mass_barge_dry) ...
    / (mass_motor_assembled + mass_barge_dry);
       
%% increased restoring moment due to heave staff
% KF is the distance from keel to roll pin
% GM is distance from center of gravity to metacenter
% FM is distance from roll axis (point at which heave staff acts
% vertically) to metacenter
KF = 0.03665;   
FM = KB + BM - KF;  
FM = 0;
GM = KB + BM - KG;
C_44 = 9.81 * ( GM*mass_barge_full - FM*9.81*(mass_heave_staff + mass_roll_mech) );

%% inertial moment arm (around roll axis)
% cog_ballast is the center of gravity of the (rack + moving masses) with
% the keel as the origin. 
% KF is the distance from the keel to the roll axis
% inertial moment arm is the distance from the center of gravity of the
% accelerating masses to the roll axis
inertial_moment_arm = cog_ballast - KF;

%% Moment of inertia
% for each major contribution to mass, compute its addition to the overall
% roll moment of inertia about the waterplane (pivot) with equation m*r^2.
motor_height = 0.042 + deck_height; % [m]
motor_mass = 0.692; % [kg]
motor_inertia = motor_mass * (motor_height - draft) ^ 2;

rack_height = 0.035 + deck_height;
rack_inertia = control_rack_mass * ( (rack_height) ^ 2 + 1 / 12 * ...
    (control_rack_length) ^ 2);

mass_height = 0.0171 + deck_height;
mass_inertia = number_control_mass_layers * 2 * single_control_mass ...
    * (0.275 ^ 2 + mass_height ^ 2);

added_inertia_roll = rack_inertia + mass_inertia + motor_inertia;

updated_inertia_roll = initial_inertia_roll + added_inertia_roll;
