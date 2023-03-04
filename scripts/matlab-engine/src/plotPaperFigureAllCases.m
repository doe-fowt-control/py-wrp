function plotPaperFigureAllCases(time_array, plant_md_history, ...
    predicted_md_history, roll_plant_cases, roll_predicted_cases, ...
    mv_cases, results_folder, ~, pred_horizon_array)
% Initialize figure
figure;
set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 0.5 1]);

% Plot data even when simulation terminated early
some_time = time_array(1 : size(plant_md_history, 2));

% Colors for lines
number_cases = size(roll_plant_cases, 1);
number_MD_lines = 2;
distinct_lines = number_cases + number_MD_lines;
line_colors = parula(distinct_lines);

% Plot measured disturbance, plant and prediction
plant_col = line_colors(1, :);
pred_col = line_colors(2, :);
for MD_data = 1 : 3
    subplot(5, 1, MD_data);
    hold on;
    plot(some_time, plant_md_history(MD_data, :), 'color', plant_col);
    plot(some_time, predicted_md_history(MD_data, :), '--', ...
        'color', pred_col);
    switch MD_data
        case 1
            title('Heave Excitation Force Disturbance');
            ylabel('$F^E_3$ [N]', 'Interpreter','latex');
        case 2
            title('Roll Excitation Force Disturbance');
            ylabel('$F^E_4$ [Nm]', 'Interpreter','latex');
        case 3
            title('Wave Elevation Disturbance');
            ylabel('$\eta$ [m]', 'Interpreter','latex');
            legend('Float', 'Prediction')
    end
end

% Plot roll angle, plant and prediction over all cases
subplot(5, 1, 4);
hold on;
for plot_case = 1 : number_cases
    color = line_colors(plot_case + 2, :);
    roll_plant = roll_plant_cases(plot_case, :);
    roll_pred = roll_predicted_cases(plot_case, :);
    hold on;
    plot(some_time, roll_plant, 'color', color);
    plot(some_time, roll_pred, '--', 'color', color);
end
title('Roll Angle');
ylabel('$\zeta_4$ [rad]', 'Interpreter','latex');
left_axes_handle = get(gca); % Label equivalent in degrees
yyaxis right;
set(gca, 'YLim', left_axes_handle.YLim);
set(gca, 'YTick', left_axes_handle.YTick);
set(gca, 'YTickLabel', ...
    num2str(rad2deg(left_axes_handle.YTick)', '%4.1f'));
ylabel('[deg]');

% Plot manipulated variable over all cases
subplot(5, 1, 5);
hold on;
for plot_case = 1 : number_cases
    color = line_colors(plot_case + 2, :);
    plot(some_time, mv_cases(plot_case, :), 'color', color);
end
xlabel('Time [s]');
ylabel('$l$ [m]', 'Interpreter','latex');
title('Controller Position');
leg_cell{1} = 'h_p = 0';
for plot_case = 1 : number_cases - 1
    %rounded_cycles = round(pred_horizon_in_cycles(plot_case + 1), 2);
    leg_case = ['h_p = ' num2str(pred_horizon_array(plot_case + 1))];
    leg_cell{plot_case + 1} = leg_case;
end
legend(leg_cell);

% Grid on for all subplot axes
axes_handle = findobj(gcf, 'type', 'axes');
set(axes_handle, 'XMinorGrid', 'on', 'YMinorGrid', 'on');

% Save plots to directory
plot_save = [results_folder '\All_Cases'];
saveas(gcf, plot_save, 'png');
saveas(gcf, plot_save, 'fig');
saveas(gcf, plot_save, 'epsc');
close(gcf);

end