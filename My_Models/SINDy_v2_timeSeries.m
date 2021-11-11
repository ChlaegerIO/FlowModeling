clear all, close all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
lambda = 0.1;                     % sparsification knob for SINDy
split = 0.8;                        % split between the train and test data

% ------------------------------------------------------------------------

%% Generate data
% ../Videos/St_fog/fog_video_above_timelapse_10x_low.mov
% ../Videos/St_fog/fog_video_near_60x_low.mov
% ../Videos/Ac_lenticularis/Ac_timelapse_sunrise_low.mov
% ../Videos/Ac_night/Ac_timelapse_night_low.mov
% ../Videos/Cb/Cb_timelapse_low.mov
% ../Videos/Ci_Cu/Ci_Cu_timelapse1_low.mov
% ../Videos/Cu/Cu_timelapse_Trim_low.mov
% ../Videos/Sc/sc_beneath_timelapse_150x_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov1_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov2_low.mov
[X1_train, X1_test, video1] = importVideo('../Videos/Cu/Cu_timelapse_Trim_low.mov',split);

X1_train = matrixToNorm(X1_train, 0, 0.8);
X1_test = matrixToNorm(X1_test, 0, 0.8);
% makeVideo('figures_SINDy_v1/Cu_timelapse_Trim', X1_train, video1.Height, video1.Width);

X1_train2 = X1_train(:, 2:301);
X1_train1 = X1_train(:, 1:300);
fprintf('video input done \n');

%% SINDy coordinates - time series for V
% do SVD and take r orthonormal POD coordinates
[U1_1,S1_1,V1_1] = svd(X1_train1,'econ');
% r = round(split*video1.NumFrames/2);
r=100;
U1_1 = U1_1(:,1:r);                     % truncate with rank r and get r coordinates
S1_1 = S1_1(1:r,1:r);
V1_1 = V1_1(:,1:r);
% do SVD and take r orthonormal POD coordinates of the next step
[U1_2,S1_2,V1_2] = svd(X1_train2,'econ');
U1_2 = U1_2(:,1:r);                     % truncate with rank r and get r coordinates
S1_2 = S1_2(1:r,1:r);
V1_2 = V1_2(:,1:r);
fprintf('coordinates done \n');

%% SINDy library - in time
ThetaV = buildTheta(V1_1,r,2);
fprintf('library done \n');

%% SINDy regression
XiV = sparsifyDynamics(ThetaV,V1_2,lambda,r);
fprintf('regression done \n');

%% prediction
% problem --> V1_pred becomes too big and dominates the frame, with the
% norm the whole image gets the same intensity --> use the ODE to predict
until = size(X1_test,2);
until = 1;
V1_pred = V1_1;
% X1_pred = U1_1*S1_1*V1_pred';
for i = 1:until   
    V1_pred = buildTheta(V1_pred,r,2)*XiV;
end


untilFrame = 50;

tspan_pred = [1:0.1:untilFrame/10];
v0_pred = V1_1(1,:);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,r));
[t,Vt_pred3]=ode45(@(t,v) cloudODE(t,v,XiV,r,2),tspan_pred,v0_pred,options);



X1_pred = U1_1*S1_1*V1_pred';
X1_pred = matrixToNorm(X1_pred, 0, 0.9);

% makeVideo('figures_SINDy_v2/Cu_timelapse_lambda0.1_pol2_startV1_pred', X1_pred, video1.Height, video1.Width);
% save ('XiV_0.1.mat', 'XiV','-v7.3');
fprintf('video ouput done \n');
