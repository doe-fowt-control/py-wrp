function y = measuredOutputFunction(x, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ...
    ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ...
    ~, ~, ~)
roll_stringpot_distance = 0.16713; % [m] distance from centerline
y = zeros(2, 1);
y(1) = x(1); % heave stringpot measurement
y(2) = tan(x(3)) * roll_stringpot_distance; % roll stringpot measurement [m] to roll angle [rad]
end