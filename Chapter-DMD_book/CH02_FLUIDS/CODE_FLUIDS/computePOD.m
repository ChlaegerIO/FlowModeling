clear all, close all, clc
load ../DATA_FLUIDS/CYLINDER_ALL.mat
X = VORTALL;
Y = [X X];

%% augment matrix with mirror images to enforce symmetry/anti-symmetry
for k=1:size(X,2)
    xflip = reshape(flipud(reshape(X(:,k),nx,ny)),nx*ny,1);
    Y(:,k+size(X,2)) = -xflip;
end

%% compute mean and subtract;
VORTavg = mean(Y,2);
f1 = plotCylinder(reshape(VORTavg,nx,ny),nx,ny);  % plot average wake

%% compute POD after subtracting mean (i.e., do PCA)
[PSI,S,V] = svd(Y-VORTavg*ones(1,size(Y,2)),'econ');
% PSI are POD modes
figure
semilogy(diag(S), 'x-', 'LineWidth',1.5); % plot singular vals

for k=1:10  % plot first four POD modes
    f1 = plotCylinder(reshape(PSI(:,k),nx,ny),nx,ny);
end