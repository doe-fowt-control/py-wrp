function [excitation_amplitude_heave, excitation_phase_heave, ...
    excitation_amplitude_roll, excitation_phase_roll] ...
    = interpolateExcitationAndResponse(number_AQWA_frequencies, ...
    AQWA_incident_wave_angles, excitation_amplitudes, ...
    excitation_phases, incident_wave_angle, density_scale, ...
    scale_squared, scale_third_power)

% Initialize output arrays
excitation_amplitude_heave = zeros(1, number_AQWA_frequencies);
excitation_phase_heave = zeros(1, number_AQWA_frequencies);
excitation_amplitude_pitch = zeros(1, number_AQWA_frequencies);
excitation_phase_pitch = zeros(1, number_AQWA_frequencies);
excitation_amplitude_roll = zeros(1, number_AQWA_frequencies);
excitation_phase_roll = zeros(1, number_AQWA_frequencies);

% Interpolation of excitation force (amplitude and phase) and linear RAO module for the specified wave angle
for n = 1 : number_AQWA_frequencies
    % Amplitude of heave excitation force (N)
    excitation_amplitude_heave(n)  = interp1(AQWA_incident_wave_angles(:), excitation_amplitudes(3, :, n), incident_wave_angle, 'spline', 'extrap') * density_scale * scale_squared; 
    % Phase of heave excitation force
    excitation_phase_heave(n)  = interp1(AQWA_incident_wave_angles(:), excitation_phases(3, :, n), incident_wave_angle, 'spline', 'extrap') * pi / 180;   
    % Amplitude of pitch excitation force (N)
    excitation_amplitude_pitch(n)  = interp1(AQWA_incident_wave_angles(:), excitation_amplitudes(5, :, n), incident_wave_angle, 'spline', 'extrap') * density_scale * scale_third_power; 
    % Phase of pitch excitation force
    excitation_phase_pitch(n)  = interp1(AQWA_incident_wave_angles(:), excitation_phases(5, :, n), incident_wave_angle, 'spline', 'extrap') * pi / 180;    
    % Amplitude of roll excitation force (N)
    excitation_amplitude_roll(n)  = interp1(AQWA_incident_wave_angles(:),excitation_amplitudes(4,:,n),incident_wave_angle,'spline','extrap')*density_scale*scale_third_power;
    % Phase of roll excitation force
    excitation_phase_roll(n)  = interp1(AQWA_incident_wave_angles(:),excitation_phases(4,:,n),incident_wave_angle,'spline','extrap')*pi/180;
end

end