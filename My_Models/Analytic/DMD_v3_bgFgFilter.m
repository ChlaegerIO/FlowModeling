close all, clear all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\Analytic\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
% used number of frames
nrOfFramesUsed = 400;

rr = 330;                   % rsvd rank
q = 1;                      % rsvd power iteration
p = 5;                      % rsvd oversampling parameter

r = rr;                    % truncate at r, look at singular values
factor = 3;                 % how long to predict the future
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
% read video
% video =  VideoReader('../Videos/train/Ac_Fabio_low.mov')
% video =  VideoReader('../Videos/train/Ac_Fabio2_low.mov')
% video =  VideoReader('../Videos/train/Ac_Fabio3_low.mov')
% video =  VideoReader('../Videos/train/Ac_Fabio4_low.mov')
% video =  VideoReader('../Videos/train/Ac_night_low.mov')
% video =  VideoReader('../Videos/train/Ac_Nov1_low.mov')
% video =  VideoReader('../Videos/train/Ac_Nov2_low.mov')
% video =  VideoReader('../Videos/train/Cb_1_low.mov')
% video =  VideoReader('../Videos/train/Cb_2_low.mov')
% video =  VideoReader('../Videos/train/Cu_1_low.mov')
video =  VideoReader('../Videos/train/Cu_2_Trim_low.mov')
% video =  VideoReader('../Videos/train/Cu_3_1_low.mov')
% video =  VideoReader('../Videos/train/Cu_3_2_low.mov')
% video =  VideoReader('../Videos/train/Cu_Fabio_low.mov')
% video =  VideoReader('../Videos/train/Cu_Fabio2_low.mov')
% video =  VideoReader('../Videos/train/Sc_1_low.mov')
% video =  VideoReader('../Videos/train/St_Fabio1_low.mov')
% video =  VideoReader('../Videos/train/St_Fabio2_low.mov')
% video =  VideoReader('../Videos/train/St_near_timelapse_low.mov')
% video =  VideoReader('../Videos/train/St_Nov21_1_low.mov')
% video =  VideoReader('../Videos/train/St_Nov21_2_Trim_low.mov')
% video =  VideoReader('../Videos/train/St_Oct_low.mov')

% define paths to save video
path_figure = 'figures_v3/Cu_2_'

nrOfFramesUsed = round(video.NumFrames - 1);
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

X = matrixToNorm(X,0, 0.8);

% print input video, only do once
videoOut_input = VideoWriter(strcat(path_figure, 'Video_Input','Grayscale AVI'))
open(videoOut_input);
for i = 1:size(X,2)
    frame_gray = reshape(X(:,i),nx,ny);
    writeVideo(videoOut_input,frame_gray);
end
close(videoOut_input);

% filter for one cloud if desired, TODO: make a function
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
    print('-djpeg', '-loose', [path_figure sprintf('window.jpeg')]);
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
x1 = X(:,1);
b = (W*eigs)^(-1)*U'*x1;         % better way of calculation b
xlast = X(:, size(X,2));

% plot singular values and Cumulative Energy
figure('Name', 'Singular values'), subplot(1,2,1)
semilogy(diag(S), 'x-', 'LineWidth',1.5), grid on
xlabel('r'), ylabel('Singular value, \sigma_r'), set(gca, 'FontSize', 14)
subplot(1,2,2)
plot(cumsum(diag(S))/sum(diag(S)), 'k', 'LineWidth',2), grid on
xlabel('r'), ylabel('Cumulative Energy'), set(gca, 'FontSize', 14)
set(gcf, 'Color', 'w', 'Position', [400 200 800 600]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [10 10 16 12], 'PaperPositionMode', 'manual');
print('-djpeg', '-loose', [path_figure sprintf('singularvalues.jpeg')]);

%  Plot DMD spectrum
figure
set(gcf,'Position',[500 100 600 400])
theta = (0:1:100)*2*pi/100;
% plot(cos(theta),sin(theta),'k--')       % plot unit circle
hold on, grid on
scatter(real(omega),imag(omega),'ok')
axis([-1.1 1.1 -1.1 1.1]);
print('-djpeg', '-loose', [path_figure sprintf('omega.jpeg')]);

%% separate foreground from background
bg = find(abs(omega)<fg_bg_epsilon);
fg = setdiff(1:r, bg);

omega_fg = omega(fg);           % foreground
Phi_fg = Phi(:,fg);             % DMD foreground modes
b_fgLast = Phi_fg\xlast;

omega_bg = omega(bg);           % background
Phi_bg = Phi(:,bg);             % DMD background mode
b_bgLast = Phi_bg\xlast;

%% plot first picInX*picInY POD modes
picInX = 6;
picInY = 4;
PODmodes = zeros(nx*picInX,ny*picInY);
count = 1;
for i=1:picInX
    for j=1:picInY
        PODmodes(1+(i-1)*nx:i*nx,1+(j-1)*ny:j*ny) = reshape(real(Phi_fg(:,picInY*(i-1)+j)),nx,ny);
        count = count + 1;
    end
end

figure, axes('position',[0  0  1  1]), axis off
imagesc(PODmodes), colormap gray
print('-djpeg', '-loose', [path_figure sprintf('Phi_fgModes.jpeg')]);

% free up space if necessary
sizeOfX = size(X,2);
% clear lambda;
% clear X2;
% clear X; 
% clear U

% TODO: could be improved with mrDMD!


%% video prediction
until = factor*sizeOfX;
time_dynamics_pred = zeros(r, until);
t = (0:until-1)*dt;                     % time vector
k = (0:until-1);                        % discrete time vector

% background prediction
X_bg = zeros(numel(omega_bg), length(t));
% for tt = 1:length(t)
%     X_bg(:, tt) = b_bgLast .* exp(omega_bg.*t(tt));
% end
% X_bg = Phi_bg * X_bg;

% spectral, discrete prediction
for kk = 1:length(k)
    X_bg(:,kk) = b_bgLast .* lambda(bg).^(kk-1);
end
X_bg = Phi_bg * X_bg;

% foreground recreation
X_fg = [X-abs(X_bg(:, 1:size(X,2)))];   % abs() nimmt auch im() Werte mit
R = zeros(size(X_bg,1), size(X_bg,2));

b_fgLast = Phi_fg\X(:,size(X,2));

for i = 1:size(X_fg,1)                  % add R to X_bg after
    for j = 1:size(X_fg,2)
        if (X_fg(i,j)) < 0
            R(i,j) = X_fg(i,j); 
            X_fg(i,j) = 0;
        end
    end
end

X_bg = X_bg + R;

figure, axes('position',[0  0  1  1]), axis off
imagesc(reshape(real(X_fg(:,10)),nx,ny)), colormap gray
print('-djpeg', '-loose', [path_figure sprintf('fg_frame10.jpeg')]);

% foreground prediction
X_fg = zeros(numel(omega_fg), length(t));
for tt = 1:length(t)
    X_fg(:, tt) = b_fgLast .* exp(omega_fg.*t(tt));
end
X_fg = Phi_fg * X_fg;

% spectral, discrete prediction
for kk = 1:length(k)
    X_fg(:,kk) = b_fgLast .* lambda(fg).^(kk-1);
end
X_fg = Phi_fg * X_fg;

X_bg = matrixToNorm(X_bg,0, 0.8);
X_fg = matrixToNorm(X_fg,0, 1);

X_dmd_pred = X_bg + X_fg;               % add both solutions together

X_dmd_pred = matrixToNorm(X_dmd_pred, 0, 1);

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
videoOut = VideoWriter(strcat(path_figure, 'factor', num2str(factor),'_rank', num2str(rr),'of', num2str(nrOfFramesUsed),'_bgFgPrediction','Grayscale AVI'))
open(videoOut);
for i = 1:size(X_dmd_pred,2)
    frame_gray_out = reshape(real(X_dmd_pred(:,i)),nx,ny);
    writeVideo(videoOut,frame_gray_out)
end
close(videoOut);

