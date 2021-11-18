clear all, close all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
lambda = 0.099;                     % sparsification knob for SINDy
split = 0.8;                        % split between the train and test data

% ------------------------------------------------------------------------

%% Generate data
% ../Videos/St_fog/fog_video_above_timelapse_fast_low.mov
% ../Videos/St_fog/fog_video_near_60x_low.mov
% ../Videos/Ac_lenticularis/Ac_timelapse_sunrise_low.mov
% ../Videos/Ac_night/Ac_timelapse_night_low.mov
% ../Videos/Cb/Cb_timelapse_low.mov
% ../Videos/Ci_Cu/Ci_Cu_timelapse1_low.mov
% ../Videos/Cu/Cu_timelapse_Trim_low.mov
% ../Videos/Sc/sc_beneath_timelapse_150x_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov1_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov1_low_short.mp4
% ../Videos/Ac_St_Cu/Ac_timelapseNov2_low.mov
[X_train, X_test, video] = importVideo('../Videos/St_fog/fog_video_above_timelapse_fast_low.mov',split);

X_train = matrixToNorm(X_train, 0, 0.8);
X_test = matrixToNorm(X_test, 0, 0.8);
makeVideo('figures_SINDy_v2/St_fog', X_train, video.Height, video.Width);

X_train1 = X_train(:, 1:100);
X_train2 = X_train(:, 2:101);
fprintf('video input done \n');

%% SINDy coordinates - time series for V
% do SVD and take r orthonormal POD coordinates
[U1_1,S1_1,V1_1] = svd(X_train1,'econ');
r = size(X_train1,2) - 5;
U1_1 = U1_1(:,1:r);                     % truncate with rank r and get r coordinates
S1_1 = S1_1(1:r,1:r);
V1_1 = V1_1(:,1:r);
% do SVD and take r orthonormal POD coordinates of the next step
[U1_2,S1_2,V1_2] = svd(X_train2,'econ');
U1_2 = U1_2(:,1:r);                     % truncate with rank r and get r coordinates
S1_2 = S1_2(1:r,1:r);
V1_2 = V1_2(:,1:r);
fprintf('coordinates done \n');

%% SINDy library - in time
ThetaV = buildTheta(V1_1,r,1);
fprintf('library done \n');

%% SINDy regression
XiV = sparsifyDynamics(ThetaV,V1_2,lambda,1);
figure
imagesc(real(XiV))
figure
imagesc(imag(XiV))
fprintf('regression done \n');

%% prediction
% problem --> V1_pred becomes too big and dominates the frame, with the
% norm the whole image gets the same intensity --> use the ODE to predict
% until = size(X_test,2);
until = 150;
V1_pred = V1_1(1,:);
for i = 1:until   
    V1_pred = [V1_pred; buildTheta(V1_pred,r,1)*XiV];
end
X_pred = U1_1*S1_1*V1_pred';


%%
% untilFrame = 10;
% step = 0.1;% tspan_pred = [1:step:untilFrame];
% v0_pred = V1_1(1,:);
% options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,r));
% [t,Vt_pred]=ode45(@(t,v) cloudODE(t,v,XiV,r,1),tspan_pred,v0_pred,options);
% 
% Vt_pred = Vt_pred(1:1/step:untilFrame/step,:);
% 
% X_pred = U1_1*S1_1*Vt_pred';
% X_pred = matrixToNorm(X_pred, 0, 0.9);

makeVideo('figures_SINDy_v2/St_fog_lambda0.99_pol1_pred1_1', X_pred, video.Height, video.Width);
% save ('St_XiV_lam0.99_f100_r1.mat', 'XiV','-v7.3');
fprintf('video ouput done \n');


% %%
% [EV, eig] = eig(XiV(1:r,:));
% figure
% plot(Vt_pred(:,3));
% writematrix(eig,'St_XiV_eigenvalues_lam0.099.txt')
