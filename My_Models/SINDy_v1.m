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
[X1_train, X1_test, video1] = importVideo('../Videos/Cu/Cu_timelapse_Trim_low.mov',split);
% [X2_train, X2_test, video2] = importVideo('../Videos/St_fog/fog_video_above_timelapse_10x_low.mov',0.8);
% [X3_train, X3_test, video3] = importVideo('../Videos/Cb/Cb_timelapse_low.mov',0.8);

X1_train = matrixToNorm(X1_train, 0.8);
X1_test = matrixToNorm(X1_test, 0.8);
% makeVideo('figures_SINDy_v1/Cu_timelapse_Trim', X1_train, video1.Height, video1.Width);

X1_train2 = X1_train(:, 2:end);
X1_train1 = X1_train(:, 1:end-1);


%% SINDy coordinates
% TODO: subtract the mean
% avgX1 = mean(X1_train1,2);           % compute average X
% X1_train1 = X1_train1 - avgX1*ones(1,size(X1_train1,2));
% X1_train1 = matrixToNorm(X1_train1, 0.8);

% % plot average picture
% figure('Name', 'average image'), axes('Position',[0 0 1 1]), axis off
% imagesc(reshape(avgX1,video1.Height,video1.Width));
% colormap gray                       % color map
% print('-djpeg', '-loose', ['figures_SINDy_v1/' sprintf('Cu_timelapse_Trim_avgImage.jpeg')]);
% % plot a picture without average
% figure('Name', 'difference image'), axes('Position',[0 0 1 1]), axis off
% imagesc(reshape(X1_train1(:,1),video1.Height,video1.Width));
% colormap gray                       % color map
% print('-djpeg', '-loose', ['figures_SINDy_v1/' sprintf('Cu_timelapse_Trim_withoutAvgImage.jpeg')]);

% do SVD and take r orthonormal POD coordinates
[U1_1,S1_1,V1_1] = svd(X1_train1,'econ');
r = round(split*video1.NumFrames/2);
U1_1 = U1_1(:,1:r);                     % truncate with rank r and get r coordinates
S1_1 = S1_1(1:r,1:r);
V1_1 = V1_1(:,1:r);
% do SVD and take r orthonormal POD coordinates of the next step
[U1_2,S1_2,V1_2] = svd(X1_train2,'econ');
r = round(split*video1.NumFrames/2);
U1_2 = U1_2(:,1:r);                     % truncate with rank r and get r coordinates
S1_2 = S1_2(1:r,1:r);
V1_2 = V1_2(:,1:r);
fprintf('coordinates done \n');

%% SINDy library
% time as external (control) variable?
ThetaU = buildTheta(U1_1,r,1);
ThetaS = buildTheta(S1_1,r,1);
ThetaV = buildTheta(V1_1,r,1);
fprintf('library done \n');

%% SINDy regression
XiU = sparsifyDynamics(ThetaU,U1_2,lambda,r);
XiS = sparsifyDynamics(ThetaS,S1_2,lambda,r);
XiV = sparsifyDynamics(ThetaV,V1_2,lambda,r);
% poolDataList({'x','y','z'},Xi,r,1);
fprintf('regression done \n');

%% prediction
until = size(X1_test,2);
U1_pred = U1_2;
S1_pred = S1_2;
V1_pred = V1_2;
for i = 1:until
    U1_pred = buildTheta(U1_pred,r,1)*XiU;
    S1_pred = buildTheta(S1_pred,r,1)*XiS;    
    V1_pred = buildTheta(V1_pred,r,1)*XiV;
end

% TODO: real prediction in a specific range
X1_pred = U1_pred*S1_pred*V1_pred';
X1_pred = matrixToNorm(X1_pred, 0.9);

makeVideo('figures_SINDy_v1/Cu_timelapse_Trim_prediction_lambda0.5_startU1_2', X1_pred, video1.Height, video1.Width);
