clear all, close all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
lambda = 0.5;                     % sparsification knob for SINDy
split = 0.8;                        % split between the train and test data

% ------------------------------------------------------------------------

%% Generate data
% ../Videos/St_fog/fog_video_above_timelapse_10x.mov
% ../Videos/St_fog/fog_video_above_timelapse_10x_low.mov
% ../Videos/St_fog/fog_video_near_60x.mov
% ../Videos/St_fog/fog_video_near_60x_low.mov
% ../Videos/Ac_lenticularis/Ac_timelapse_sunrise.mp4
% ../Videos/Ac_lenticularis/Ac_timelapse_sunrise_low.mov
% ../Videos/Ac_night/Ac_timelapse_night.mp4
% ../Videos/Ac_night/Ac_timelapse_night_low.mov
% ../Videos/Cb/Cb_timelapse.mov
% ../Videos/Cb/Cb_timelapse_low.mov
% ../Videos/Ci_Cu/Ci_Cu_timelapse1.mp4
% ../Videos/Ci_Cu/Ci_Cu_timelapse1_low.mov
% ../Videos/Cu/Cu_timelapse_Trim.mp4
% ../Videos/Cu/Cu_timelapse_Trim_low.mov
% ../Videos/Sc/sc_beneath_timelapse_150x.mov
% ../Videos/Sc/sc_beneath_timelapse_150x_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov1.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov2.mov
[X1_train, X1_test, video1] = importVideo('../Videos/St_fog/fog_video_above_timelapse_10x_low.mov',split);

X1_train = matrixToNorm(X1_train, 0.8);
X1_test = matrixToNorm(X1_test, 0.8);
% makeVideo('figures_SINDy_v1/Cu_timelapse_Trim', X1_train, video1.Height, video1.Width);

X1_train2 = X1_train(:, 2:101);
X1_train1 = X1_train(:, 1:100);
fprintf('video input done \n');

%% SINDy coordinates - time series for V
% do SVD and take r orthonormal POD coordinates
[U1_1,S1_1,V1_1] = svd(X1_train1,'econ');
r = round(split*video1.NumFrames/2);
r=90;
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
until = size(X1_test,2);
until = 1;
V1_pred = V1_1;
for i = 1:until   
    V1_pred = buildTheta(V1_pred,r,2)*XiV;
end

% TODO: real prediction in a specific range
X1_pred = U1_1*S1_1*V1_pred';
X1_pred = matrixToNorm(X1_pred, 0.9);

makeVideo('figures_SINDy_v2/St_fog_above_lambda0.5_startV1_1', X1_pred, video1.Height, video1.Width);
fprintf('video ouput done \n');
