clear all, close all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
lambda = 0.099;                     % sparsification knob for SINDy
split = 0.8;                        % split between the train and test data

% ------------------------------------------------------------------------

%% Generate data
% ../Videos/St_fog/fog_video_above_timelapse_10x_low.mov
% ../Videos/St_fog/fog_video_above_timelapse_fast_low.mov
% ../Videos/St_fog/fog_video_near_60x_low.mov
% ../Videos/Ac_lenticularis/Ac_timelapse_sunrise_low.mov
% ../Videos/Ac_night/Ac_timelapse_night_low.mov
% ../Videos/Cb/Cb_timelapse_low.mov
% ../Videos/Ci_Cu/Ci_Cu_timelapse1_low.mov
% ../Videos/Cu/Cu_timelapse_Trim_low.mov
% ../Videos/Sc/sc_beneath_timelapse_150x_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov1_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov2_low.mov
[X1_train, X1_test, video1] = importVideo('../Videos/St_fog/fog_video_above_timelapse_fast_low.mov',split);

X1_train = matrixToNorm(X1_train, 0, 0.8);
X1_test = matrixToNorm(X1_test, 0, 0.8);
% makeVideo('figures_SINDy_v1/Cu_timelapse_Trim', X1_train, video1.Height, video1.Width);

X1_train2 = X1_train(:, 2:101);
X1_train1 = X1_train(:, 1:100);
fprintf('video input done \n');

%% SINDy coordinates - time series for V
r = 99;
[U,S,V] = svd(X1_train1,'econ');
U = U(:,1:r);                   % truncate with rank r
S = S(1:r,1:r);
V = V(:,1:r);   
Atilde = U'*X1_train2*V*inv(S);
[W,eigs] = eig(Atilde);
Phi = X1_train2*V*inv(S)*W;

lambda_eig = diag(eigs);            % discrete-time eigenvalues
omega = log(lambda_eig);         % continuous-time eigenvalues

% make b as coordinates
rCoord = 5;
for ii=1:(rCoord+1)
    b(:,ii) = Phi\X1_train1(:, ii);
end
fprintf('coordinates done \n');

%% SINDy library - in time
Thetab = buildTheta(b(:,1:rCoord)',r,1);
fprintf('library done \n');

%% SINDy regression
Xib = sparsifyDynamics(Thetab,b(:,2:(rCoord+1))',lambda,1);
figure
imagesc(real(Xib))
figure
imagesc(imag(Xib))
fprintf('regression done \n');

%% prediction
untilFrame = 50;
step = 0.1;
tspan_pred = [1:step:untilFrame];
b0_pred = b(:,1)';
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,r));
[t,b_pred]=ode45(@(t,b) cloudODE(t,b,Xib,r,1),tspan_pred,b0_pred,options);

b_pred = b_pred(1:1/step:untilFrame/step,:);

until = 130;
time_dynamics_pred = zeros(r, until);
t = (0:until-1);                     % time vector
for iter = 1:until
    time_dynamics_pred(:,iter) = (b_pred(untilFrame,:)'.*exp(omega*t(iter)));
end

X1_pred_dmd = Phi * time_dynamics_pred;

X1_pred_dmd = matrixToNorm(X1_pred_dmd, 0, 0.9);

makeVideo('figures_SINDy_v3/St_lambda0.99_pol1_ODE', X1_pred_dmd, video1.Height, video1.Width);
save ('Data/St_Xib_lam0.99_f100_r1_v3.mat', 'Xib','-v7.3');
fprintf('video ouput done \n');

