clear all, close all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

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
[X1_train, X1_test, video1] = importVideo('../Videos/Cu/Cu_timelapse_Trim_low.mov',0.8);

X1_train = matrixToNorm(X1_train, 0.8);
X1_test = matrixToNorm(X1_test, 0.8);

makeVideo('figures_SINDy_v1/Cu_timelapse_Trim', X1_train, video1.Height, video1.Width);


%% SINDy coordinates


%% SINDy 

