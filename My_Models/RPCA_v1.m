close all, clear all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% tuning parameters
% ------------------------------------------------------------------------
% used number of frames
nrOfFramesUsed = 100;

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
video = VideoReader('../Videos/Cu/Cu_timelapse_Trim_low.mov')
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

% TODO: Koopman

%%  Compute RPCA
[L,S] = RPCA(X);

%%
subplot(2,2,1)
imagesc(reshape(X(:,10),nx,ny)), colormap gray
subplot(2,2,3)
imagesc(reshape(L(:,10),nx,ny)), colormap gray
subplot(2,2,4)
imagesc(reshape(S(:,10),nx,ny)), colormap gray
print('-djpeg', '-loose', ['figures_RPCA_v1/' sprintf('Cu_timelapse_Trim.jpeg')]);


