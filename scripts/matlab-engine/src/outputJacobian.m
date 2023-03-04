function C = outputJacobian(x, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ...
    ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ...
    ~, ~, ~)
roll_stringpot_distance = 0.16713; % [m] distance from centerline
C = zeros(2, length(x));
C(1, 1) = 1;
C(2, 3) = roll_stringpot_distance * sec(x(3))^2;
end