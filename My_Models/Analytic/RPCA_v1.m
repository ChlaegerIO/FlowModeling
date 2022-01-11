close all, clear all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\Analytic\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
% used number of frames
nrOfFramesUsed = 10;

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
path_figure = 'figures_RPCA_v1/Cu_2_'

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

X = matrixToNorm(X, 0,0.8);

clear frame, clear frame_gray;  % free up space

% create input data matrix
X2 = X(:,2:end);
X = X(:,1:end-1);

%%  Compute RPCA
[L,S] = RPCA(X);

%%
subplot(2,2,1)
imagesc(reshape(X(:,5),nx,ny)), colormap gray
subplot(2,2,3)
imagesc(reshape(L(:,5),nx,ny)), colormap gray
subplot(2,2,4)
imagesc(reshape(S(:,5),nx,ny)), colormap gray
print('-djpeg', '-loose', [path_figure sprintf('RPCA.jpeg')]);


