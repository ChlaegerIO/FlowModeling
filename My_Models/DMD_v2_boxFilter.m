close all, clear all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
% used number of frames
nrOfFramesUsed = 50;

r = 49;                    % truncate at r, look at singular values
factor = 2;                 % how long to predict the future
dt = 1;                     % timesteps

withBox = false;            % True: with the box filter, False: without
%    ___________
%   |           |
% x |           |
%   |___________|
%         y
% cl1_posx = 0;              % position left upper corner of rectancle for cloud 1 in pixel
% cl1_posy = 0;
% cl1_sizex = 480;            % size of rectangle in pixels
% cl1_sizey = 720;

cl1_posx = 40;              % position left upper corner of rectancle for cloud 1 in pixel
cl1_posy = 115;
cl1_sizex = 280;            % size of rectangle in pixels
cl1_sizey = 220;

% ------------------------------------------------------------------------

%% input data/video processing
% TODO: change to lower video quality
% read video
% ../Videos/St_fog_real_timelapse/fog_video_above_timelapse_10x.mov
% ../Videos/St_fog_real_timelapse/fog_video_above_timelapse_10x_low.mov
% ../Videos/St_fog_real_timelapse/fog_video_near_60x.mov
% ../Videos/St_fog_real_timelapse/fog_video_near_60x_low.mov
% ../Videos/Ac_lenticularis_timelapse_sunrise_short/Ac_timelapse_sunrise.mp4
% ../Videos/Ac_lenticularis_timelapse_sunrise_short/Ac_timelapse_sunrise_low.mov
% ../Videos/Ac_timelapse_night/Ac_timelapse_night.mp4
% ../Videos/Ac_timelapse_night/Ac_timelapse_night_low.mov
% ../Videos/Cb_timelapse/Cb_timelapse.mov
% ../Videos/Cb_timelapse/Cb_timelapse_low.mov
% ../Videos/Ci_Cu_timelapse/Ci_Cu_timelapse1.mp4
% ../Videos/Ci_Cu_timelapse/Ci_Cu_timelapse1_low.mov
% ../Videos/Cu_timelapse/Cu_timelapse_Trim.mp4
% ../Videos/Cu_timelapse/Cu_timelapse_Trim_low.mov
% ../Videos/Sc_real_timelapse/sc_beneath_timelapse_150x.mov
% ../Videos/Sc_real_timelapse/sc_beneath_timelapse_150x_low.mov
video = VideoReader('../Videos/Cu_timelapse/Cu_timelapse_Trim_low.mov')
% nrOfFramesUsed = round(video.NumFrames - 1);
nx = video.Height;
ny = video.Width;
row = nx*ny;
X = zeros(row, nrOfFramesUsed);
ii = 1;
while hasFrame(video) && ii <= nrOfFramesUsed 
    frame = readFrame(video);
    frame_gray = double(rgb2gray(frame));
    X(:,ii) = reshape(frame_gray,[row, 1]);
    ii = ii + 1;
end

% kind of a norm of X to [0,1]
X = X - min(X(:));
X = X ./max(X(:));
X = X ./1.2;                % darken image, especially highlights

% % print input video, only do once
% videoOut_input = VideoWriter('figures_v2/Cu_timelapse_Trim','Grayscale AVI')
% open(videoOut_input);
% for i = 1:size(X,2)
%     frame_gray = reshape(X(:,i),nx,ny);
%     writeVideo(videoOut_input,frame_gray);
% end
% close(videoOut_input);

avgX = mean(X,2);           % compute average X in [0,1]
X = X - avgX*ones(1,size(X,2));     % now between [-1,1]
X = X - min(X(:));          % shift image again to [0,1]
X = X ./max(X(:));
X = X ./1.2;

% % plot average picture
% figure('Name', 'average image'), axes('Position',[0 0 1 1]), axis off
% imagesc(reshape(avgX,nx,ny));
% colormap gray               % color map
% print('-djpeg', '-loose', ['figures_v2/' sprintf('Cu_timelapse_Trim_avgImage.jpeg')]);

% % print input minus avg video
% videoOut_input = VideoWriter('figures_v2/Cu_timelapse_Trim_minus_avg_Video_Input','Grayscale AVI')
% open(videoOut_input);
% for i = 1:size(X,2)
%     frame_gray = reshape(X(:,i),nx,ny);
%     writeVideo(videoOut_input,frame_gray);
% end
% close(videoOut_input);

% filter for one cloud
if withBox == true
    filter = zeros(size(X,1),1);
    for ii = 1:size(filter)
        % in y direction
        if ii >= (nx*cl1_posy) && ii <= nx*(cl1_posy + cl1_sizey)
            % in x direction
            if mod(ii,nx) >= cl1_posx && mod(ii,nx) <= (cl1_posx + cl1_sizex)
                filter(ii) = 1;
            end
        end
    end
    X = X.*(filter*ones(1,size(X,2)));
    % figure;
    % imagesc(reshape(X(:,1),nx,ny));
    % colormap gray
    % print('-djpeg', '-loose', ['figures_v2/' sprintf('Cu_timelapse_Trim_window.jpeg')]);
end

clear frame, clear frame_gray;  % free up space

% create input data matrix
X2 = X(:,2:end);
X = X(:,1:end-1);

% TODO: Koopman
% TODO: only focus on one object at a time


%%  Compute DMD
[U,S,V] = svd(X,'econ');
U = U(:,1:r);                   % truncate with rank r
S = S(1:r,1:r);
V = V(:,1:r);   
Atilde = U'*X2*V*inv(S);
[W,eigs] = eig(Atilde);
Phi = X2*V*inv(S)*W;

lambda = diag(eigs);            % discrete-time eigenvalues
omega = log(lambda)/dt;            % continuous-time eigenvalues
x1 = X(:, 1);
% b = Phi\x1;
b = (W*eigs)^(-1)*U'*x1;         % better way of calculation b


% % plot singular values and Cumulative Energy
% figure('Name', 'Singular values'), subplot(1,2,1)
% semilogy(diag(S), 'x-', 'LineWidth',1.5), grid on
% xlabel('r')
% ylabel('Singular value, \sigma_r')
% set(gca, 'FontSize', 14)
% subplot(1,2,2)
% plot(cumsum(diag(S))/sum(diag(S)), 'k', 'LineWidth',2), grid on
% xlabel('r');
% ylabel('Cumulative Energy')
% set(gca, 'FontSize', 14)
% set(gcf, 'Color', 'w', 'Position', [400 200 800 600]);
% set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [10 10 16 12], 'PaperPositionMode', 'manual');
% print('-djpeg', '-loose', ['figures_v2/' sprintf('Cu_timelapse_Trim_singularvalues_allframes.jpeg')]);

% %  Plot DMD spectrum
% figure
% set(gcf,'Position',[500 100 600 400])
% theta = (0:1:100)*2*pi/100;
% plot(cos(theta),sin(theta),'k--')       % plot unit circle
% hold on, grid on
% scatter(real(diag(eigs)),imag(diag(eigs)),'ok')
% axis([-1.1 1.1 -1.1 1.1]);
% print('-djpeg', '-loose', ['figures_v2/' sprintf('Cu_timelapse_Trim_eigenvalues_allframes.jpeg')]);


% %% plot first 24 POD modes
% PODmodes = zeros(nx*6,ny*4);
% count = 1;
% for i=1:6
%     for j=1:4
%         PODmodes(1+(i-1)*nx:i*nx,1+(j-1)*ny:j*ny) = reshape(U(:,4*(i-1)+j),nx,ny);
%         count = count + 1;
%     end
% end
% 
% figure, axes('position',[0  0  1  1]), axis off
% imagesc(PODmodes), colormap gray
% print('-djpeg', '-loose', ['figures_v2/' sprintf('Cu_timelapse_Trim_PODmodes_allframes.jpeg')]);

% free up space if necessary
sizeOfX = size(X,2);
clear lambda, clear X2, clear x;
% clear X; 
% clear U

% TODO: could be improved with mrDMD!


%% video reconstruction and prediction
until = factor*sizeOfX;
time_dynamics_pred = zeros(r, until);
t = (0:until-1)*dt;                     % time vector
for iter = 1:until
    time_dynamics_pred(:,iter) = (b.*exp(omega*t(iter)));
end

% %%
% X_dmd_pred in [0,2] after added average
X_dmd_pred = Phi * time_dynamics_pred;
X_dmd_pred = X_dmd_pred + avgX*ones(1,size(time_dynamics_pred,2));

% shift image again to [0,1]
X_dmd_pred = matrixToNorm(X_dmd_pred, 0,1);

% if some values are < 0 or > 1
if min(X_dmd_pred(:)) < 0 || max(X_dmd_pred(:)) > 1
    for i = 1:size(X_dmd_pred,1)
        for j = 1:size(X_dmd_pred,2)
            if real(X_dmd_pred(i,j)) < 0
                X_dmd_pred(i,j) = 0;
            elseif real(X_dmd_pred(i,j)) > 1
                X_dmd_pred(i,j) = 1;
            end
        end
    end
end

% recreate and make a prediction as a video
videoOut = VideoWriter('figures_v2/Cu_timelapse_Trim_prediction_out_factor2_50frames_r=744','Grayscale AVI')
open(videoOut);
for i = 1:size(X_dmd_pred,2)
    frame_gray_out = reshape(real(X_dmd_pred(:,i)),nx,ny);
    writeVideo(videoOut,frame_gray_out)
end
close(videoOut);

