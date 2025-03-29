clear; clc;

%% System Setup
T = 100;                     % Time steps
n = 2; m = 1;                % State and control dimensions
p1 = 1; p2 = 1;              % Vision & proprioception dimensions

A = [1 0.01; -1 0.98];       % Dynamics: simple arm model
B = [0; 1];                  % Control affects velocity

% Observation models: combined position & velocity in vision
H1 = [1 1];                  % Vision: both position & velocity
H2 = [0 1];                  % Proprioception: velocity only

r = [1; 0];                  % Target: hold at position 1, zero velocity

Q_kalman = 0.01 * eye(n);    % Process noise
R1 = 0.01; R2 = 0.02;        % Sensor noise (vision, proprioception)
P = eye(n);                  % Initial covariance for Kalman filter

% LQR (volitional control)
Q = 1 * eye(n);
R = 10 * eye(m);
[P_lqr, ~, ~] = idare(A, B, Q, R);
K_lqr = inv(B' * P_lqr * B + R) * (B' * P_lqr * A);

% Haptic LQR (external assistive control)
scale = 10;
[P_hap, ~, ~] = idare(A, B, scale*Q, scale*R);
K_hap = inv(B' * P_hap * B + scale*R) * (B' * P_hap * A);

% Feedforward gain for tracking
S = H1 * ((eye(n) - A + B * K_lqr) \ B);
N = pinv(S);  % For volitional controller
Nhap = pinv(H1 * ((eye(n) - A + B * K_hap) \ B));  % For assistive controller

HypnoGain = 10;  % Suggestion-like mixing gain amplification

%% Initialization
x = zeros(n, T+1);           % True state
xhat = zeros(n, T+1);        % Estimated state
u = zeros(m, T);             % Total control input
y1 = zeros(p1, T);           % Vision
y2 = zeros(p2, T);           % Proprioception

rng(1);  % For reproducibility

%% Simulation Loop
for t = 2:T
    % --- External Haptic Assistive Control (e.g., XR force field)
    u_ext = -K_hap * x(:,t) + Nhap * r(1);

    % --- Sensor feedback with noise
    y1(:,t) = H1 * x(:,t) + sqrt(R1) * randn;
    y2(:,t) = H2 * x(:,t) + sqrt(R2) * randn;
    y_combined = [y1(:,t); y2(:,t)];
    H_combined = [H1; H2];
    R_combined = diag([R1 R2]);

    % --- Kalman filter (belief updating)
    S_kf = H_combined * P * H_combined' + R_combined;
    K_kf = P * H_combined' / S_kf;

    % Simulate deafferentation (FD): reduced sensory integration
    K_kf = 0.01 * K_kf;

    % Simulate hypnotic suggestion: amplified internal correction
    K_kf = HypnoGain * K_kf;

    % Update internal state estimate
    innovation = y_combined - H_combined * xhat(:,t);
    xhat(:,t) = xhat(:,t) + K_kf * innovation;

    % --- Volitional Control (internal prediction-based)
    u(:,t) = -K_lqr * xhat(:,t) + N * r(1);

    % --- Effort-Reward Miscalibration
    threshold = 2 + 0.5 * randn;
    if norm(u(:,t)) > threshold
        u(:,t) = 0;  % Stops trying if perceived effort is too high
    end

    % --- Voluntary motor suppression (loss of agency)
    if t > 30
        u(:,t) = 0;  % Subject stops trying (e.g., internal inhibition)
    end

    % --- Apply total control (volitional + external haptic)
    u(:,t) = u(:,t) + u_ext;

    % --- Update true state (add process noise)
    w = sqrt(Q_kalman) * randn(n,1);
    x(:,t+1) = A * x(:,t) + B * u(:,t) + w;

    % --- Involuntary movements (e.g., dystonia/tremor)
    bias = [0.01 * sin(0.2*t); 0];
    x(:,t+1) = x(:,t+1) + bias;

    % --- Predict next belief state
    xhat(:,t+1) = A * xhat(:,t) + B * u(:,t);

    % --- Update Kalman filter covariance
    P = A * (P - K_kf * H_combined * P) * A' + Q_kalman;
end

%% Plotting Results

% --- Position and Velocity Tracking
figure;
subplot(2,1,1);
plot(0:T, x(1,:), 'k', 0:T, xhat(1,:), 'k--', 0:T, r(1)*ones(1,T+1), 'k:', 'LineWidth', 2);
legend('True Position', 'Estimated Position', 'Target');
ylabel('x_1 (Position)'); title('Position Tracking with Internal & External Control');
grid on;

subplot(2,1,2);
plot(0:T, x(2,:), 'k', 0:T, xhat(2,:), 'k--', 'LineWidth', 2);
legend('True Velocity', 'Estimated Velocity');
xlabel('Time'); ylabel('x_2 (Velocity)');
grid on;

% --- Control Input Over Time
figure;
plot(1:T, u, 'k', 'LineWidth', 2);
xlabel('Time Step');
ylabel('u(t)');
title('Total Control Input (LQG + Haptic + Pathological Modulators)');
grid on;
