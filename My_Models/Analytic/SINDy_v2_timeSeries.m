clear all, close all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
lambda = 0.099;                     % sparsification knob for SINDy
split = 0.8;                        % split between the train and test data
polyOrder = 1;

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
path_figure = 'figures_SINDy_v2/Cu_2_'

X_train = matrixToNorm(X_train, 0, 0.8);
X_test = matrixToNorm(X_test, 0, 0.8);
makeVideo(strcat(path_figure,'Video_Input'), X_train, video.Height, video.Width);

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
ThetaV = buildTheta(V1_1,r,polyOrder);
fprintf('library done \n');

%% SINDy regression
XiV = sparsifyDynamics(ThetaV,V1_2,lambda,polyOrder);
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
    V1_pred = [V1_pred; buildTheta(V1_pred(size(V1_pred,1),:),r,1)*XiV];
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
X_pred = matrixToNorm(X_pred, 0, 0.9);

makeVideo(strcat(path_figure,'lambda',num2str(lambda),'_pol',num2str(polyOrder),'_pred'), X_pred, video.Height, video.Width);
% save ('St_XiV_lam0.99_f100_r1.mat', 'XiV','-v7.3');
fprintf('video ouput done \n');


% %%
% [EV, eig] = eig(XiV(1:r,:));
% figure
% plot(Vt_pred(:,3));
% writematrix(eig,'St_XiV_eigenvalues_lam0.099.txt')
