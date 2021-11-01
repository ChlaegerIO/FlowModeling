close all, clear all, clc

%% tuning parameters
% ------------------------------------------------------------------------
% used number of frames
nrOfFramesUsed = 300;

rr = 298;                   % rsvd rank
q = 1;                      % rsvd power iteration
p = 5;                      % rsvd oversampling parameter

r = 298;                    % truncate at r, look at singular values
factor = 2;                 % how long to predict the future
dt = 1;                     % timesteps width
fg_bg_epsilon = 1e-2;       % foreground background separation, omegas around +-epsilon of origin

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

X = matrixToNorm(X, 0.8);

% print input video, only do once
% videoOut_input = VideoWriter('figures_v3/Cu_timelapse_Trim','Grayscale AVI')
% open(videoOut_input);
% for i = 1:size(X,2)
%     frame_gray = reshape(X(:,i),nx,ny);
%     writeVideo(videoOut_input,frame_gray);
% end
% close(videoOut_input);

% filter for one cloud if desired
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
    figure;
    imagesc(reshape(X(:,1),nx,ny));
    colormap gray
    print('-djpeg', '-loose', ['figures_v3/' sprintf('Cu_timelapse_Trim_window.jpeg')]);
end

clear frame, clear frame_gray;  % free up space

% create input data matrix
X2 = X(:,2:end);
X = X(:,1:end-1);

% TODO: Koopman

%%  Compute DMD
[U,S,V] = svd(X,'econ');
U = U(:,1:r);                   % truncate with rank r
S = S(1:r,1:r);
V = V(:,1:r);   
Atilde = U'*X2*V*inv(S);
[W,eigs] = eig(Atilde);
Phi = X2*V*inv(S)*W;

lambda = diag(eigs);            % discrete-time eigenvalues
omega = log(lambda)/dt;         % continuous-time eigenvalues
xlast = X(:, size(X,2));

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
% print('-djpeg', '-loose', ['figures_v3/' sprintf('Cu_timelapse_Trim_singularvalues_300frames.jpeg')]);

%  Plot DMD spectrum
figure
set(gcf,'Position',[500 100 600 400])
theta = (0:1:100)*2*pi/100;
% plot(cos(theta),sin(theta),'k--')       % plot unit circle
hold on, grid on
scatter(real(omega),imag(omega),'ok')
axis([-1.1 1.1 -1.1 1.1]);
print('-djpeg', '-loose', ['figures_v3/' sprintf('Cu_timelapse_Trim_omega_300frames.jpeg')]);

%% separate foreground from background
bg = find(abs(omega)<fg_bg_epsilon);
fg = setdiff(1:r, bg);

omega_fg = omega(fg);           % foreground
Phi_fg = Phi(:,fg);             % DMD foreground modes
b_fgLast = Phi_fg\xlast;

omega_bg = omega(bg);           % background
Phi_bg = Phi(:,bg);             % DMD background mode
b_bgLast = Phi_bg\xlast;

% %% plot first picInX*picInY POD modes
% picInX = 6;
% picInY = 4;
% PODmodes = zeros(nx*picInX,ny*picInY);
% count = 1;
% for i=1:picInX
%     for j=1:picInY
%         PODmodes(1+(i-1)*nx:i*nx,1+(j-1)*ny:j*ny) = reshape(real(Phi_fg(:,picInY*(i-1)+j)),nx,ny);
%         count = count + 1;
%     end
% end
% 
% figure, axes('position',[0  0  1  1]), axis off
% imagesc(PODmodes), colormap gray
% print('-djpeg', '-loose', ['figures_v3/' sprintf('Cu_timelapse_Trim_Phi_fg_modes_300frames.jpeg')]);

% free up space if necessary
sizeOfX = size(X,2);
clear lambda;
% clear X2;
% clear X; 
% clear U

% TODO: could be improved with mrDMD!


%% video prediction
until = factor*sizeOfX;
time_dynamics_pred = zeros(r, until);
t = (0:until-1)*dt;                     % time vector

% background prediction
X_bg = zeros(numel(omega_bg), length(t));
for tt = 1:length(t)
    X_bg(:, tt) = b_bgLast .* exp(omega_bg.*t(tt));
end
X_bg = Phi_bg * X_bg;

% % foreground recreation
% X_fg = [X-abs(X_bg(:, 1:size(X,2)))];   % abs() nimmt auch im() Werte mit
% R = zeros(size(X_bg,1), size(X_bg,2));
% 
% b_fgLast = Phi_fg\X(:,size(X,2));
% 
% for i = 1:size(X_fg,1)                  % add R to X_bg after
%     for j = 1:size(X_fg,2)
%         if (X_fg(i,j)) < 0
%             R(i,j) = X_fg(i,j); 
%             X_fg(i,j) = 0;
%         end
%     end
% end
% 
% X_bg = X_bg + R;

% foreground prediction
X_fg = zeros(numel(omega_fg), length(t));
for tt = 1:length(t)
    X_fg(:, tt) = b_fgLast .* exp(omega_fg.*t(tt));
end
X_fg = Phi_fg * X_fg;

X_bg = matrixToNorm(X_bg, 0.5);
X_fg = matrixToNorm(X_fg, 1);

X_dmd_pred = X_bg + X_fg;               % add both solutions together

X_dmd_pred = matrixToNorm(X_dmd_pred, 1);

% if some values are < 0 or > 1
if min(real(X_dmd_pred(:))) < 0 || max(real(X_dmd_pred(:))) > 1
    for i = 1:size(X_dmd_pred,1)
        if (min(real(X_dmd_pred(i,:))) >= 0 && max(real(X_dmd_pred(i,:))) <= 1)
            continue                    % skip this line
        end
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
videoOut = VideoWriter('figures_v3/Cu_timelapse_Trim_factor2_300frames_r=298_bg_fg_only_prediction','Grayscale AVI')
open(videoOut);
for i = 1:size(X_dmd_pred,2)
    frame_gray_out = reshape(real(X_dmd_pred(:,i)),nx,ny);
    writeVideo(videoOut,frame_gray_out)
end
close(videoOut);

