clear all, close all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\Analytic\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
lambda = 0.5;                     % sparsification knob for SINDy
split = 0.8;                        % split between the train and test data

% ------------------------------------------------------------------------

%% Generate data
% [X_train, X_test, video] =  importVideo('../Videos/train/Ac_Fabio_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Ac_Fabio2_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Ac_Fabio3_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Ac_Fabio4_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Ac_night_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Ac_Nov1_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Ac_Nov2_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Cb_1_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Cb_2_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Cu_1_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Cu_2_Trim_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Cu_3_1_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Cu_3_2_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Cu_Fabio_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Cu_Fabio2_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/Sc_1_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/St_Fabio1_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/St_Fabio2_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/St_near_timelapse_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/St_Nov21_1_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/St_Nov21_2_Trim_low.mov',split);
% [X_train, X_test, video] =  importVideo('../Videos/train/St_Oct_low.mov',split);

% define paths to save video
path_figure = 'figures_SINDy_v1/Cu_2_'

X_train = matrixToNorm(X_train, 0, 0.8);
X_test = matrixToNorm(X_test, 0, 0.8);
makeVideo(strcat(path_figure, 'Video_Input'), X_train, video.Height, video.Width);

X_train1 = X_train(:, 1:100);
X_train2 = X_train(:, 2:101);
fprintf('video input done \n');

%% SINDy coordinates
% TODO: subtract the mean
% avgX1 = mean(X_train1,2);           % compute average X
% X_train1 = X_train1 - avgX1*ones(1,size(X_train1,2));
% X_train1 = matrixToNorm(X_train1, 0,0.8);

% % plot average picture
% figure('Name', 'average image'), axes('Position',[0 0 1 1]), axis off
% imagesc(reshape(avgX1,video1.Height,video1.Width));
% colormap gray                       % color map
% print('-djpeg', '-loose', [path_figure sprintf('avgImage.jpeg')]);
% % plot a picture without average
% figure('Name', 'difference image'), axes('Position',[0 0 1 1]), axis off
% imagesc(reshape(X1_train1(:,1),video1.Height,video1.Width));
% colormap gray                       % color map
% print('-djpeg', '-loose', [path_figure sprintf('withoutAvgImage.jpeg')]);

% do SVD and take r orthonormal POD coordinates
[U1_1,S1_1,V1_1] = svd(X_train1,'econ');
r = size(X_train1,2) - 2;
U1_1 = U1_1(:,1:r);                     % truncate with rank r and get r coordinates
S1_1 = S1_1(1:r,1:r);
V1_1 = V1_1(:,1:r);
% do SVD and take r orthonormal POD coordinates of the next step
[U1_2,S1_2,V1_2] = svd(X_train2,'econ');
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
fprintf('regression XiU done \n');
XiS = sparsifyDynamics(ThetaS,S1_2,lambda,r);
fprintf('regression XiS done \n');
XiV = sparsifyDynamics(ThetaV,V1_2,lambda,r);
fprintf('regression done \n');

%% prediction
until = size(X_test,2);

until = 100;
U1_pred = U1_2;
S1_pred = S1_2;
V1_pred = V1_2;
for i = 1:until
    U1_pred = buildTheta(U1_pred,r,2)*XiU;
    S1_pred = buildTheta(S1_pred,r,2)*XiS;
    V1_pred = buildTheta(V1_pred,r,2)*XiV;
end

%% TODO: real prediction in a specific range
Ut = U1_pred;
St = S1_pred;
% [Ut,St,Vt] = svd(X_test,'econ');
% Ut = Ut(:,1:r);
% St = St(1:r,1:r);

X1_pred = Ut*St*V1_pred';
avgX1 = mean(X_test,2);           % compute average X
X1_pred = X1_pred + 2*avgX1*ones(1,size(X1_pred,2));
X1_pred = matrixToNorm(X1_pred, 0, 0.9);

makeVideo(strcat(path_figure,'lambda',num2str(lambda),'_until',num2str(until),'_St_Ut'), X1_pred, video.Height, video.Width);
fprintf('prediction done \n');
