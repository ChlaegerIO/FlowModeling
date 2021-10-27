close all, clear all;

%% Define time and space discretizations
n = 200;
m = 80;
x = linspace(-15,15,n);
t = linspace(0,8*pi,m); 
dt = t(2) - t(1); 
[Xgrid,T] = meshgrid(x,t);

%% Create two spatio-temporal patterns
f1 = 0.5*cos(Xgrid) .* (1+0*T);                     % time-independent!
f2 = (sech(Xgrid).*tanh(Xgrid)) .* (2*exp(1j*2.8*T));

%% Combine signals and make data matrix
X = (f1 + f2)';                                     % Data Matrix

figure;
subplot(2,3,1);
surfl(real(f1)); 
shading interp; colormap(gray); view(-20,60);
title('f1(x,t)')

subplot(2,3,2);
surfl(real(f2)); 
shading interp; colormap(gray); view(-20,60);
title('f2(x,t)')

subplot(2,3,3);
surfl(real(X')); 
shading interp; colormap(gray); view(-20,60);
title('f(x,t)')

%% Create data matrices for DMD
X1 = X(:,1:end-1);
X2 = X(:,2:end);

%% SVD and rank-50 truncation
r = 50;                                             % rank truncation
[U, S, V] = svd(X1, 'econ');
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

%% Build Atilde and DMD Modes
Atilde = Ur'*X2*Vr/Sr;
[W, D] = eig(Atilde);
Phi = X2*Vr/Sr*W;                                   % DMD Modes

%% DMD Spectra
fg_bg_epsilon = 1e-2;
lambda = diag(D);
omega = log(lambda)/dt;

bg = find(abs(omega)<fg_bg_epsilon);
fg = setdiff(1:r, bg);

omega_fg = omega(fg);                               % foreground
Phi_fg = Phi(:,fg);                                 % DMD foreground modes

omega_bg = omega(bg);                               % background
Phi_bg = Phi(:,bg);                                 % DMD background mode

%% Compute DMD Foreground Solution
b = Phi_bg \ X(:, 1);
X_bg = zeros(numel(omega_bg), length(t));
for tt = 1:length(t),
    X_bg(:, tt) = b .* exp(omega_bg .* t(tt));
end;
X_bg = Phi_bg * X_bg;
% X_bg = X_bg(1:n, :);

subplot(2,3,4);
surfl(real(X_bg')); 
shading interp; colormap(gray); view(-20,60);
title('x_{bg}(t)')

%% Compute DMD Background Solution
b = Phi_fg \ X(:, 1);
X_fg = zeros(numel(omega_fg), length(t));
for tt = 1:length(t),
    X_fg(:, tt) = b .* exp(omega_fg .* t(tt));
end;
X_fg = Phi_fg * X_fg;
X_fg(:,1) = X_fg(:,1) - X_bg(:,1);                  % remove background in first state
% X_fg = X_fg(1:n, :);


subplot(2,3,5);
surfl(real(X_bg'));surfl(real(X_fg')); 
shading interp; colormap(gray); view(-20,60);
title('x_{fg}(t)')

%% plot both background and foreground together
X_dmd = X_bg +X_fg;

subplot(2,3,6);
surfl(real(X_bg'));surfl(real(X_dmd'));
shading interp; colormap(gray); view(-20,60);
title('x_{dmd}(t)')

print('-djpeg', '-loose', ['figures/' sprintf('example.jpeg')]);


%% plot omegas
figure;
plot(omega, '.');
xlabel('Re');
ylabel('Im');
print('-djpeg', '-loose', ['figures/' sprintf('example_omegas.jpeg')]);

