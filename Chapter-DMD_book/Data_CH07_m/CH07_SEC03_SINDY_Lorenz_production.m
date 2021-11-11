clear all, close all, clc
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3;
x0=[-8; 8; 27];  % Initial condition
tspan=[.01:.01:50];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,3));
[t,x]=ode45(@(t,x) lorenz(t,x,Beta),tspan,x0,options);

%% Compute Derivative
for i=1:length(x)
    dx(i,:) = lorenz(0,x(i,:),Beta);
end

%% Pool Data  (i.e., build library of nonlinear time series)
polyorder = 3;
Theta = poolData(x,n,polyorder);
m = size(Theta,2);

figure;
plot3(x(:,1),x(:,2),x(:,3))

%% Compute Sparse regression: sequential least squares
lambda = 0.025;      % lambda is our sparsification knob.
Xi = sparsifyDynamics(Theta,dx,lambda,n)
poolDataLIST({'x','y','z'},Xi,n,polyorder);

%% Predict the future
% can't predict the future like this --> the value blows up.
Theta_last = poolData(x(1,:),n,polyorder);
for i = 1:10000
    x_pred(i,:) = Theta_last*Xi;
    Theta_last = poolData(x_pred(i,:),n,polyorder);
end

% figure;
% plot3(x_pred(:,1),x_pred(:,2),x_pred(:,3))

% new approach by using the dynamics I received
x_pred2 = poolData(x,n,polyorder)*Xi;
% figure;
% plot3(x_pred2(:,1),x_pred2(:,2),x_pred2(:,3))

tspan_pred = [.1:.01:100];
x0_pred = [8.5; 8.5; 27.5];
[t_pred3,x_pred3]=ode45(@(t,x) lorenz(t,x,Beta),tspan_pred,x0_pred,options);

figure;
plot3(x_pred3(:,1),x_pred3(:,2),x_pred3(:,3))

