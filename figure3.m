clear; clc;

%% Parameters
T = 60;                      % Total time steps
n = 2; m = 1; p = 1;
omega = 2 * pi * 10;         % 10 Hz oscillatory dynamics

A = 0.1 * [0, 0.01; -omega^2, 0.98];
B = [0; 1];
U = [1, 0];                  % Observation matrix
Q_kalman = 0.01 * eye(n);
P = eye(n);                  % Initial prior covariance
G = repmat({[5.0 1.0]}, 1, T);  % Feedback gain

% Phase switch points
self_hypnosis_end = 20;
simulation_only_end = 40;

%% Initialization
x     = zeros(n, T+1); x(:,1) = [0; 0];   % True state
xhat  = zeros(n, T+1);                    % Estimated state
xbar  = zeros(n, T+1);                    % Prior predicted state
y     = zeros(p, T);                      % Sensory input
yhat  = zeros(p, T);                      % Predicted sensory output
u     = zeros(m, T);
innovation = zeros(p, T);
K_kf  = cell(1, T);
S_ext = [0.1; 0];                         % External suggestion bias

%% Simulation
for t = 1:T
    % ------- Stage 1: With Sensory Input ---------
    if t <= self_hypnosis_end
        R = 0.01;                                   % Trusted sensory input
        G_t = diag([1]);                            % Gating matrix: full attention
        y(:,t) = U * x(:,t) + sqrt(R) * randn;      % Noisy measurement

        yhat(:,t) = U * xhat(:,t);
        innovation(:,t) = y(:,t) - yhat(:,t);
        S = U * P * U' + R;
        K_kf{t} = P * U' / S;

        xhat(:,t) = xhat(:,t) + K_kf{t} * G_t * innovation(:,t);

        % ------- Stage 2: Internal Simulation (No Sensory Input) -----
    elseif t <= simulation_only_end
        R = 0.001;                      % Noise variance
        G_t = diag([1]);                % Gating matrix: full attention
        y(:,t) = sqrt(R) * randn;       % No measurement, just noise

        yhat(:,t) = U * xhat(:,t);
        innovation(:,t) = y(:,t) - yhat(:,t);
        S = U * P * U' + R;
        K_kf{t} = P * U' / S;

        xhat(:,t) = xhat(:,t) + K_kf{t} * G_t * innovation(:,t);

        % ------- Stage 3: Internal Simulation with External Suggestion -----
    else
        % Add bias term S_ext to internal dynamics
        R = 0.000001;                           % High trust with Hypnotic Induction 
        G_t = diag([1]);                        % Gating matrix: full attention
        y(:,t) = S_ext(1)+ sqrt(R) * randn;     % Not sensed (no reafference) but imagined (exafference) with Hypnotic suggestion

        yhat(:,t) = U * xhat(:,t);
        innovation(:,t) = y(:,t) - yhat(:,t);
        S = U * P * U' + R;
        K_kf{t} = P * U' / S;

        xhat(:,t) = xhat(:,t) + K_kf{t} * G_t * innovation(:,t);

    end

    % Control from estimate
    u(:,t) = -G{t} * xhat(:,t);

    % True dynamics with process noise
    w = sqrt(Q_kalman) * randn(n,1);
    x(:,t+1) = A * x(:,t) + B * u(:,t) + w;

    % Predict next estimate (prior)
    xhat(:,t+1) = A * xhat(:,t) + B * u(:,t);

    % % Update covariance
    % P = A * (P - K_kf{t} * U * P) * A' + Q_kalman;
    if ~isempty(K_kf{t})
        P = A * (P - K_kf{t} * U * P) * A' + Q_kalman;
    else
        P = A * P * A' + Q_kalman;  % No correction applied
    end
end

%% Plot Prediction Error
figure;
plot(1:T, innovation(1,:), 'k', 'LineWidth', 2); hold on;
xline(self_hypnosis_end, 'r--', 'LineWidth', 2, 'Label', 'Insight Meditation Begin');
xline(simulation_only_end, 'g--', 'LineWidth', 2, 'Label', 'Hypnotic Suggestions Begin');
xlabel('Time Step');
ylabel('Prediction Error y(t) - ŷ(t)');
title('Prediction Error Across Phases');
legend('Innovation', 'Insight Meditation Begin', 'Hypnotic Suggestions Begin');
grid on;

%% Animation of True vs Estimated x₁
figure;
h1 = plot(NaN, NaN, 'k', 'LineWidth', 2); hold on;
h2 = plot(NaN, NaN, 'r--', 'LineWidth', 2);
xline(self_hypnosis_end, 'r--', 'LineWidth', 2, 'Label', 'Insight Meditation Begin');
xline(simulation_only_end, 'g--', 'LineWidth', 2, 'Label', 'Hypnotic Suggestions Begin');
xlabel('Time Step'); ylabel('x_1');
title('True vs Estimated State x₁(t)');
legend('True x₁', 'Estimated x₁');
xlim([0 T]);
ylim([-0.5 0.5]);
grid on;

for t = 1:T
    set(h1, 'XData', 0:t, 'YData', x(1,1:t+1));
    set(h2, 'XData', 0:t, 'YData', xhat(1,1:t+1));
    drawnow;
    pause(0.05);
end
