close all, clear all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\Analytic\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
% used number of frames
nrOfFramesUsed = 200;

r = 190;                    % truncate at r, look at singular values
factor = 4;                 % how long to predict the future
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
% read video
% video =  VideoReader('../Videos/train/Ac_Fabio_low.mov')
% video =  VideoReader('../Videos/train/Ac_Fabio2_low.mov')
% video =  VideoReader('../Videos/train/Ac_Fabio3_low.mov')
% video =  VideoReader('../Videos/train/Ac_Fabio4_low.mov')
video =  VideoReader('../Videos/train/Ac_night_low.mov')
% video =  VideoReader('../Videos/train/Ac_Nov1_low.mov')
% video =  VideoReader('../Videos/train/Ac_Nov2_low.mov')
% video =  VideoReader('../Videos/train/Cb_1_low.mov')
% video =  VideoReader('../Videos/train/Cb_2_low.mov')
% video =  VideoReader('../Videos/train/Cu_1_low.mov')
% video =  VideoReader('../Videos/train/Cu_2_Trim_low.mov')
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
path_figure = 'figures_v2/Ac_night_'

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
X = matrixToNorm(X,0, 0.9);

% print input video, only do once
videoOut_input = VideoWriter(strcat(path_figure, 'Video_Input'),'Grayscale AVI')
open(videoOut_input);
for i = 1:size(X,2)
    frame_gray = reshape(X(:,i),nx,ny);
    writeVideo(videoOut_input,frame_gray);
end
close(videoOut_input);

avgX = mean(X,2);           % compute average X in [0,1]
X = X - avgX*ones(1,size(X,2));     % now between [-1,1]
X = matrixToNorm(X,0, 0.9);

% plot average picture
figure('Name', 'average image'), axes('Position',[0 0 1 1]), axis off
imagesc(reshape(avgX,nx,ny));
colormap gray               % color map
print('-djpeg', '-loose', [path_figure sprintf('avgImage.jpeg')]);

% print input minus avg video
videoOut_input = VideoWriter(strcat(path_figure, 'minus_avg_Video_Input'),'Grayscale AVI')
open(videoOut_input);
for i = 1:size(X,2)
    frame_gray = reshape(X(:,i),nx,ny);
    writeVideo(videoOut_input,frame_gray);
end
close(videoOut_input);

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
    figure;
    imagesc(reshape(X(:,1),nx,ny));
    colormap gray
    print('-djpeg', '-loose', [path_figure sprintf('window.jpeg')]);
end

clear frame, clear frame_gray;  % free up space

% create input data matrix
X2 = X(:,2:end);
X = X(:,1:end-1);


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
b = Phi\x1;

% plot singular values and Cumulative Energy
figure('Name', 'Singular values'), subplot(1,2,1)
semilogy(diag(S), 'x-', 'LineWidth',1.5), grid on
xlabel('r')
ylabel('Singular value, \sigma_r')
set(gca, 'FontSize', 14)
subplot(1,2,2)
plot(cumsum(diag(S))/sum(diag(S)), 'k', 'LineWidth',2), grid on
xlabel('r');
ylabel('Cumulative Energy')
set(gca, 'FontSize', 14)
set(gcf, 'Color', 'w', 'Position', [400 200 800 600]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [10 10 16 12], 'PaperPositionMode', 'manual');
print('-djpeg', '-loose', [path_figure sprintf('singularvalues.jpeg')]);

%  Plot DMD spectrum
figure
set(gcf,'Position',[500 100 600 400])
theta = (0:1:100)*2*pi/100;
plot(cos(theta),sin(theta),'k--')       % plot unit circle
hold on, grid on
scatter(real(diag(eigs)),imag(diag(eigs)),'ok')
axis([-1.1 1.1 -1.1 1.1]);
print('-djpeg', '-loose', [path_figure sprintf('eigenvalues.jpeg')]);


%% plot first 24 POD modes
PODmodes = zeros(nx*6,ny*4);
count = 1;
for i=1:6
    for j=1:4
        PODmodes(1+(i-1)*nx:i*nx,1+(j-1)*ny:j*ny) = reshape(U(:,10*(i-1)+j),nx,ny);
        count = count + 1;
    end
end

figure, axes('position',[0  0  1  1]), axis off
imagesc(PODmodes), colormap gray
print('-djpeg', '-loose', [path_figure sprintf(strcat('PODmodes_Box',num2str(withBox),'.jpeg'))]);

% free up space if necessary
sizeOfX = size(X,2);
clear lambda, clear X2, clear x;
% clear X; 
% clear U


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
videoOut = VideoWriter(strcat(path_figure, 'Video_predictionFactor', num2str(factor),'_rank',num2str(r),'of',num2str(nrOfFramesUsed),'_Box',num2str(withBox)),'Grayscale AVI')
open(videoOut);
for i = 1:size(X_dmd_pred,2)
    frame_gray_out = reshape(real(X_dmd_pred(:,i)),nx,ny);
    writeVideo(videoOut,frame_gray_out)
end
close(videoOut);

