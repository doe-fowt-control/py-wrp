function [excitation_amplitudes, excitation_phases, AQWA_frequencies, ...
    AQWA_incident_wave_angles, number_AQWA_frequencies, ...
    added_mass_infinity, added_mass_frequency, damping_frequency, ...
    RAO_phases, number_AQWA_angles, RAO_amplitudes] = reformatAQWAData()
    
% Linear FK+diffraction excitation force amplitude and phase: 
% excitation_force_amplitudes(degrees of freedom, incident wave angle, incident wave frequency),
% excitation_force_phases(degrees of freedom, incident wave angle, incident wave frequency).
% Linear RAO amplitude and phase: 
% RAO_amplitudes(degrees of freedom, incident wave angle, incident wave frequency),
% RAO_phases(degrees of freedom, incident wave angle, incident wave frequency).

% Load AWQA results data structure
load('AQWA_Data.mat', 'data');  
number_AQWA_frequencies= size(data, 2);
number_AQWA_angles  = size(getfield(data, {1}, 'angle'), 2);

% Initialize reformatted data matrices
degrees_of_freedom = 6; 
AQWA_frequencies = zeros(1, number_AQWA_frequencies);
AQWA_incident_wave_angles = zeros(1, number_AQWA_angles);
added_mass_infinity = zeros(1, degrees_of_freedom);
added_mass_frequency = zeros(degrees_of_freedom, degrees_of_freedom, number_AQWA_frequencies);
damping_frequency = zeros(degrees_of_freedom, degrees_of_freedom, number_AQWA_frequencies);
excitation_amplitudes = zeros(degrees_of_freedom, number_AQWA_angles, number_AQWA_frequencies); 
excitation_phases = zeros(degrees_of_freedom, number_AQWA_angles, number_AQWA_frequencies);
RAO_amplitudes = zeros(degrees_of_freedom, number_AQWA_angles, number_AQWA_frequencies);
RAO_phases = zeros(degrees_of_freedom, number_AQWA_angles, number_AQWA_frequencies);

% Loop over AQWA incident wave frequencies (for excitation force) or
% forcing frequencies (for added mass)
for n = 1 : number_AQWA_frequencies
    AQWA_frequencies(n) = getfield(data, {n}, 'omega');
    added_mass_matrix   = getfield(data, {n}, 'added_mass');
    damping_matrix   = getfield(data, {n}, 'damping');
    
    % Reformat added mass and damping matrices
    for i = 1 : degrees_of_freedom
        for j = 1 : degrees_of_freedom
            added_mass_frequency(i, j, n) = added_mass_matrix(i,j);
            damping_frequency(i, j, n) = damping_matrix(i, j);
        end
    end
    
    % Loop over AQWA incident wave angles
    AQWA_nested_data = getfield(data, {n}, 'angle');
    for m = 1 : number_AQWA_angles
        AQWA_incident_wave_angles(m) = getfield(AQWA_nested_data, {m}, 'degrees');  
        
        % Reformat excitation force (comprised of Froude-Krylov force + diffraction force)
        fkd = getfield(AQWA_nested_data, {m}, 'fkdiff'); 
        excitation_amplitudes(1, m, n) = fkd.x_amp;
        excitation_amplitudes(2, m, n) = fkd.y_amp;
        excitation_amplitudes(3, m, n) = fkd.z_amp;
        excitation_amplitudes(4, m, n) = fkd.rx_amp;
        excitation_amplitudes(5, m, n) = fkd.ry_amp;
        excitation_amplitudes(6, m, n) = fkd.rz_amp;
        excitation_phases(1, m, n) = fkd.x_phase;
        excitation_phases(2, m, n) = fkd.y_phase;
        excitation_phases(3, m, n) = fkd.z_phase;
        excitation_phases(4, m, n) = fkd.rx_phase;
        excitation_phases(5, m, n) = fkd.ry_phase;
        excitation_phases(6, m, n) = fkd.rz_phase;
        
        % Reformat RAO (vessel motion to incident wave amplitude)
        RAO = getfield(AQWA_nested_data, {m}, 'RAO'); 
        RAO_amplitudes(1, m, n) = RAO.x_amp;
        RAO_amplitudes(2, m, n) = RAO.y_amp;
        RAO_amplitudes(3, m, n) = RAO.z_amp;
        RAO_amplitudes(4, m, n) = RAO.rx_amp;
        RAO_amplitudes(5, m, n) = RAO.ry_amp;
        RAO_amplitudes(6, m, n) = RAO.rz_amp;
        RAO_phases(1, m, n) = RAO.x_phase;
        RAO_phases(2, m, n) = RAO.y_phase;
        RAO_phases(3, m, n) = RAO.z_phase;
        RAO_phases(4, m, n) = RAO.rx_phase;
        RAO_phases(5, m, n) = RAO.ry_phase;
        RAO_phases(6, m, n) = RAO.rz_phase;
     end    
end

% Estimate infinite frequency diagonal added mass values
for i = 1 : degrees_of_freedom
    added_mass_infinity(i) = added_mass_frequency(i, i, number_AQWA_frequencies);
end

end
