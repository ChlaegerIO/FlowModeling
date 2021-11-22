clear all, close all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
lambda = 0.1;                     % sparsification knob for SINDy
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

[X1_train, X1_test, video1] = importVideo('../Videos/Cu/Cu_timelapse_Trim_low.mov',split);
[X2_train, X2_test, video2] = importVideo('../Videos/Ac_St_Cu/Ac_timelapseNov1_low_short.mp4',split);
[X3_train, X3_test, video3] = importVideo('../Videos/Ci_Cu/Ci_Cu_timelapse1_low.mov',split);
[X4_train, X4_test, video4] = importVideo('../Videos/St_fog/fog_video_above_timelapse_fast_low.mov',split);
% [X5_train, X5_test, video5] = importVideo('../Videos/Ac_lenticularis/Ac_timelapse_sunrise_low.mov',split);
[X6_train, X6_test, video6] = importVideo('../Videos/Sc/sc_beneath_timelapse_150x_low.mov',split);
[X7_train, X7_test, video7] = importVideo('../Videos/Cb/Cb_timelapse_low.mov',split);
X1_train = matrixToNorm(X1_train, 0, 0.8);
X1_test = matrixToNorm(X1_test, 0, 0.8);
X2_train = matrixToNorm(X2_train, 0, 0.8);
X2_test = matrixToNorm(X2_test, 0, 0.8);
X3_train = matrixToNorm(X3_train, 0, 0.8);
X3_test = matrixToNorm(X3_test, 0, 0.8);
X4_train = matrixToNorm(X4_train, 0, 0.8);
X4_test = matrixToNorm(X4_test, 0, 0.8);
% X5_train = matrixToNorm(X5_train, 0, 0.8);
% X5_test = matrixToNorm(X5_test, 0, 0.8);
X6_train = matrixToNorm(X6_train, 0, 0.8);
X6_test = matrixToNorm(X6_test, 0, 0.8);
X7_train = matrixToNorm(X7_train, 0, 0.8);
X7_test = matrixToNorm(X7_test, 0, 0.8);

X_train1 = X1_train(:, 1:100);
X_train2 = X1_train(:, 2:101);
X_train1 = [X_train1; X2_train(:, 1:100)];
X_train2 = [X_train2; X2_train(:, 2:101)];
X_train1 = [X_train1; X3_train(:, 1:100)];
X_train2 = [X_train2; X3_train(:, 2:101)];
X_train1 = [X_train1; X4_train(:, 1:100)];
X_train2 = [X_train2; X4_train(:, 2:101)];
% X_train1 = [X_train1; X5_train(:, 1:100)];
% X_train2 = [X_train2; X5_train(:, 2:101)];
X_train1 = [X_train1; X6_train(:, 1:100)];
X_train2 = [X_train2; X6_train(:, 2:101)];
X_train1 = [X_train1; X7_train(:, 1:100)];
X_train2 = [X_train2; X7_train(:, 2:101)];

X_train1 = [X_train1; X2_train(:, 201:300)];
X_train2 = [X_train2; X2_train(:, 202:301)];
X_train1 = [X_train1; X6_train(:, 111:210)];
X_train2 = [X_train2; X6_train(:, 112:211)];
X_train1 = [X_train1; X7_train(:, 251:350)];
X_train2 = [X_train2; X7_train(:, 252:351)];
X_train1 = [X_train1; X7_train(:, 361:460)];
X_train2 = [X_train2; X7_train(:, 362:461)];
X_train1 = [X_train1; X4_train(:, 251:350)];
X_train2 = [X_train2; X4_train(:, 252:351)];
X_train1 = [X_train1; X6_train(:, 611:710)];
X_train2 = [X_train2; X6_train(:, 612:711)];
X_train1 = [X_train1; X3_train(:, 111:210)];
X_train2 = [X_train2; X3_train(:, 112:211)];
X_train1 = [X_train1; X3_train(:, 211:310)];
X_train2 = [X_train2; X3_train(:, 212:311)];
X_train1 = [X_train1; X6_train(:, 501:600)];
X_train2 = [X_train2; X6_train(:, 502:601)];
X_train1 = [X_train1; X2_train(:, 321:420)];
X_train2 = [X_train2; X2_train(:, 322:421)];
X_train1 = [X_train1; X1_train(:, 201:300)];
X_train2 = [X_train2; X1_train(:, 202:301)];
X_train1 = [X_train1; X1_train(:, 301:400)];
X_train2 = [X_train2; X1_train(:, 302:401)];
X_train1 = [X_train1; X7_train(:, 101:200)];
X_train2 = [X_train2; X7_train(:, 102:201)];
X_train1 = [X_train1; X6_train(:, 221:320)];
X_train2 = [X_train2; X6_train(:, 222:321)];
X_train1 = [X_train1; X2_train(:, 441:540)];
X_train2 = [X_train2; X2_train(:, 442:541)];
X_train1 = [X_train1; X1_train(:, 421:520)];
X_train2 = [X_train2; X1_train(:, 422:521)];
X_train1 = [X_train1; X6_train(:, 401:500)];
X_train2 = [X_train2; X6_train(:, 402:501)];
fprintf('video input done \n');

%% SINDy coordinates - time series for V
% % do SVD and take r orthonormal POD coordinates
[U1_1,S1_1,V1_1] = svd(X_train1,'econ');
r = size(X_train1,2) - 1;
U1_1 = U1_1(:,1:r);                     % truncate with rank r and get r coordinates
S1_1 = S1_1(1:r,1:r);
V1_1 = V1_1(:,1:r);
% do SVD and take r orthonormal POD coordinates of the next step
[U1_2,S1_2,V1_2] = svd(X_train2,'econ');
U1_2 = U1_2(:,1:r);                     % truncate with rank r and get r coordinates
S1_2 = S1_2(1:r,1:r);
V1_2 = V1_2(:,1:r);
fprintf('coordinates done \n');

%% plot time series
figure, set(gcf,'position',[100,50,1000,700])
for i=1:6
    for j=1:4
        VNr = (i-1)*4+j;
        subplot(6,4,VNr)
        plot(V1_1(:,VNr))
        text(20,0.25, sprintf('V %i', VNr))
    end
end
figure, set(gcf,'position',[100,50,1000,700])
for i=1:6
    for j=1:4
        VNr = (i-1)*4+j;
        subplot(6,4,VNr)
        plot(V1_2(:,VNr))
        text(20,0.25, sprintf('V %i', VNr))
    end
end
figure, set(gcf,'position',[110,70,1000,700])
for i=1:6
    for j=1:4
        VNr = (i-1)*4+j;
        subplot(6,4,VNr)
        plot(V1_1(:,VNr)-V1_2(:,VNr))
        text(20,0.25, sprintf('V %i', VNr))
    end
end
% print('-djpeg', '-loose', ['figures_SINDy_v4/' sprintf('V_St_timeSeries_train17.jpeg')]);

%% SINDy library - in time
ThetaV = buildTheta(V1_1,r,1,'trigonometric');
fprintf('library done \n');

%% SINDy regression
XiV = sparsifyDynamics(ThetaV,V1_2,lambda,r);
fprintf('regression done \n');

%% prediction
% prediction based on test set
[Utest,Stest,Vtest] = svd(X4_test,'econ');
Utest = Utest(:,1:r);                     % truncate with rank r and get r coordinates
Stest = Stest(1:r,1:r);
Vtest = Vtest(:,1:r);

untilFrame = 100;
% ODE prediction
step = 0.01;
tspan_pred = [1:step:untilFrame];
v0_pred = V1_2(1,:);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,size(Vtest,2)));
[t,Vt_pred]=ode45(@(t,v) cloudODE(t,v,XiV,r,1,'trigonometric'),tspan_pred,v0_pred,options);
Vt_pred = Vt_pred(1:1/step:untilFrame/step,:);

Xt_pred = Utest*Stest*Vt_pred';
avgX2 = mean(X4_test,2);           % compute average X
Xt_pred = Xt_pred + 2*avgX2*ones(1,size(Xt_pred,2));
Xt_pred = matrixToNorm(Xt_pred, 0, 0.9);

makeVideo('figures_SINDy_v4/St_pred_lambda0.1_sin_1x_test', Xt_pred, video1.Height, video1.Width);
fprintf('video ouput done \n');

%%
% time series of the prediction
figure, set(gcf,'position',[150,50,1000,700])
for i=1:6
    for j=1:4
        VNr = (i-1)*4+j;
        subplot(6,4,VNr)
        plot(Vt_pred(:,VNr))
        text(20,0.25, sprintf('V %i', VNr))
    end
end
print('-djpeg', '-loose', ['figures_SINDy_v4/' sprintf('St_pred_train17_lambda0.1_sin.jpeg')]);

figure, set(gcf,'position',[50,20,800,400])
imagesc(real(XiV))

% [EV, eig] = eig(XiV(1:r,:));
% writematrix(eig,'figures_SINDy_v4/XiV_eigenvalues_lam0.099.csv')
% writematrix(XiV,'figures_SINDy_v4/XiV_lam0.029.csv')
