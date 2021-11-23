clear all, close all, clc
load ../DATA_FLUIDS/CYLINDER_ALL.mat
X = VORTALL(:,1:end-1);
X2 = VORTALL(:,2:end);
[U,S,V] = svd(X,'econ');

%%  Compute DMD (Phi are eigenvectors)
r = 21;                                 % truncate at 21 modes
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);
Atilde = U'*X2*V*inv(S);
[W,eigs] = eig(Atilde);
Phi = X2*V*inv(S)*W;

lambda = diag(eigs);                    % discrete-time eigenvalues
omega = log(lambda);                    % continuous-time eigenvalues
x = X(:, 1);
b = Phi\x;

%% Plot DMD modes
for i=9:2:20
    plotCylinder(reshape(real(Phi(:,i)),nx,ny),nx,ny);
    plotCylinder(reshape(imag(Phi(:,i)),nx,ny),nx,ny);
end

% Create one graph with all together
plotInOnePlot(13,'DMD modes',1,12,"DMD");
print('-djpeg', '-loose', ['figures/' sprintf('DMD-modes1.jpeg')]);


%%  Plot DMD spectrum
figure
set(gcf,'Position',[500 100 600 400])
theta = (0:1:100)*2*pi/100;
plot(cos(theta),sin(theta),'k--')       % plot unit circle
hold on, grid on
scatter(real(diag(eigs)),imag(diag(eigs)),'ok')
axis([-1.1 1.1 -1.1 1.1]);
print('-djpeg', '-loose', ['figures/' sprintf('eigenvalues1.jpeg')]);

%% prediction
factor = 2;                             % factor to advance time
until = factor*size(X, 2);
time_dynamics_pred = zeros(r, until);
t = (0:until-1);                        % time vector
for iter = 1:until
    time_dynamics_pred(:,iter) = (b.*exp(omega*t(iter)));
end
X_dmd_pred = Phi * time_dynamics_pred;

videoCylinderFlow(real(X_dmd_pred),size(X_dmd_pred,2), nx, ny)