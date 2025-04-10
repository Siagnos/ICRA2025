clear; clc;

%% Parameters
T = 60;
n = 2; m = 1; p = 1;

%% Define the Oscillatory Plant (10 Hz)
omega = 2*pi*10;             % Angular frequency for 10 Hz oscillation
A = 0.1*[0    0.01;
    -omega^2 0.98];
B = [0; 1];
H = [1 0];

Q_kalman = 0.01 * eye(n);
P = eye(n);                  % Initial covariance

G = repmat({[5.0 1.0]}, 1, T);  % Fixed feedback gain

switch_point1 = 20;          % Start suboptimal Kalman gain
switch_point2 = 40;          % Hypnotherapy: restore

%% Initialization
x = zeros(n, T+1); x(:,1) = [0; 0];   % True state
xhat = zeros(n, T+1);                 % Estimated state
y = zeros(p, T);                      % Measured output
yhat = zeros(p, T);                   % Predicted output
innovation = zeros(p, T);             % y - yhat
u = zeros(m, T);
K_kf = cell(1, T);
S_ext = [0.1; 0];                        % External suggestion bias

%% Simulation Loop
for t = 1:T
    if t < switch_point1
        R_kalman = 0.01; % high sensory trust
        % Measurement
        y(:,t) = H * x(:,t) + sqrt(R_kalman) * randn;

        % Prediction
        yhat(:,t) = H * xhat(:,t);
        innovation(:,t) = y(:,t) - yhat(:,t);

        % Kalman Gain (dynamic switch + hypnotherapy)
        S = H * P * H' + R_kalman;
        K_kf{t} = P * H' / S;              % Optimal
    elseif t < switch_point2
        R_kalman = 10; % low sensory trust
        % Measurement
        y(:,t) = H * x(:,t) + sqrt(R_kalman) * randn;

        % Prediction
        yhat(:,t) = H * xhat(:,t);
        innovation(:,t) = y(:,t) - yhat(:,t);

        % Kalman Gain (dynamic switch + hypnotherapy)
        S = H * P * H' + R_kalman;
        K_kf{t} = (P * H' / S);     % Suboptimal (simulate sensory dysfunction)
    else
        R_kalman = 0.000001; % High trust with Hypnotic Induction 
        % Measurement
        % y(:,t) = H * x(:,t) + sqrt(R_kalman) * randn;
        y(:,t) = S_ext(1) + sqrt(R_kalman) * randn;

        % Prediction
        yhat(:,t) = H * xhat(:,t);
        innovation(:,t) = y(:,t) - yhat(:,t);

        % Kalman Gain (dynamic switch + hypnotherapy)
        S = H * P * H' + R_kalman;
        K_kf{t} = P * H' / S;              % Restored after hypnotherapy
    end

    % Update estimate with innovation
    xhat(:,t) = xhat(:,t) + K_kf{t} * innovation(:,t);

    % Control input (based on estimated state)
    u(:,t) = -G{t} * xhat(:,t);

    % Evolve true system
    w = sqrt(Q_kalman) * randn(n,1);
    x(:,t+1) = A * x(:,t) + B * u(:,t) + w;

    % Predict next estimate
    xhat(:,t+1) = A * xhat(:,t) + B * u(:,t);

    % Update covariance
    P = A * (P - K_kf{t} * H * P) * A' + Q_kalman;
end

%% Plot Prediction Error
figure;
plot(1:T, innovation(1,:), 'k', 'LineWidth', 2); hold on;
xline(switch_point1, 'r--', 'LineWidth', 2, 'Label', 'FND Dysfunction Onset');
xline(switch_point2, 'g--', 'LineWidth', 2, 'Label', 'Hypnotic Suggestions Begin');
xlabel('Time Step');
ylabel('Prediction Error y(t) - ŷ(t)');
title('Innovation (Prediction Error)');
legend('Innovation', 'FND Dysfunction Onset', 'Restoration via Hypnotic Suggestions');
grid on;

%% Animation of True vs Estimated x1
figure;
h1 = plot(NaN, NaN, 'k', 'LineWidth', 2); hold on;
h2 = plot(NaN, NaN, 'r--', 'LineWidth', 2);
xline(switch_point1, 'r--', 'LineWidth', 2, 'Label', 'FND Dysfunction Onset');
xline(switch_point2, 'g--', 'LineWidth', 2, 'Label', 'Hypnotic Suggestions Begin');
xlabel('Time Step'); ylabel('x_1');
title('True vs Estimated State x_1(t)');
legend('True x_1', 'Estimated x_1');
xlim([0 T]); ylim([-1 1]);
grid on;

for t = 1:T
    set(h1, 'XData', 0:t, 'YData', x(1,1:t+1));
    set(h2, 'XData', 0:t, 'YData', xhat(1,1:t+1));
    drawnow;
    pause(0.05);
end
