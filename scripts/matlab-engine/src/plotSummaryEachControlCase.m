function plotSummaryEachControlCase(time_array, states_history, ...
    state_estimates_history, plant_md_history, predicted_md_history, ...
    mv_history, results_folder, pred_horizon_in_cycles, ...
    pred_horizon_array, control_case)

% Plot data even when simulation terminated early
some_time = time_array(1 : size(states_history, 2));

% Plot heave and roll states
figure;
set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
for subplot_data = 1 : 4
    subplot(4, 2, subplot_data);
    plot(some_time, states_history(subplot_data, :));
    hold on;
    plot(some_time, state_estimates_history(subplot_data, :), '--');
    xlabel('Time [s]');
    switch subplot_data
        case 1
            title('Heave Position');
            ylabel('z [m]');
        case 2
            title('Heave Velocity');
            ylabel('dz/dt [m/s]');
        case 3
            title('Roll Angle');
            ylabel('\alpha [rad]');
            % Label equivalent in degrees
            left_axes_handle = get(gca);
            yyaxis right;
            set(gca, 'YLim', left_axes_handle.YLim);
            set(gca, 'YTick', left_axes_handle.YTick);
            set(gca, 'YTickLabel', ...
                num2str(rad2deg(left_axes_handle.YTick)', '%4.1f'));
            ylabel('[deg]');
        case 4
            title('Roll Angle Derivative');
            ylabel('d\alpha/dt [rad/s]');
    end
end

% Plot measured input disturbances
for subplot_data = 5 : 7
    subplot(4, 2, subplot_data);
    plot(some_time, plant_md_history(subplot_data - 4, :));
    hold on;
    plot(some_time, predicted_md_history(subplot_data - 4, :), '--');
    xlabel('Time [s]');
    switch subplot_data
        case 5
            title('Heave Excitation Force Disturbance');
            ylabel('F_{ex, 3} [N]');
        case 6
            title('Roll Excitation Force Disturbance');
            ylabel('F_{ex, 4} [Nm]');
        case 7
            title('Wave Elevation Disturbance');
            ylabel('\eta [m]');
    end
end

% Plot manipulated variable
subplot(4, 2, 8);
plot(some_time, mv_history);
xlabel('Time [s]');
ylabel('u_1 [m]');
if control_case == 1
    title('Controller Position: No Control');
elseif control_case == 2
    title('Controller Position: Control Without Previewing');
else
    l_pos = ['Controller Position With Prediction and Control' ...
        ' Horizons:'];
    num1 = num2str(pred_horizon_in_cycles(control_case));
    cycles = 'cyles';
    num2 = num2str(pred_horizon_array(control_case));
    steps = 'steps';
    l_pos_string = [l_pos ' ' num1 ' ' cycles ' (' num2 ' ' steps ')'];
    title(l_pos_string);
end

% Grid on for all subplot axes
axes_handle = findobj(gcf, 'type', 'axes');
set(axes_handle, 'XMinorGrid', 'on', 'YMinorGrid', 'on');

% Save plots to directory
plot_save = [results_folder '\Control_Case_' num2str(control_case)];
saveas(gcf, plot_save, 'png');
saveas(gcf, plot_save, 'fig');
saveas(gcf, plot_save, 'epsc');
close(gcf);

end