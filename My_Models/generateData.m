clear all, close all;
%% Generate the data and store it that this step can be skiped
% ../Videos/St_fog/fog_video_above_timelapse_10x_low.mov
% ../Videos/St_fog/fog_video_near_60x_low.mov
% ../Videos/Ac_lenticularis/Ac_timelapse_sunrise_low.mov
% ../Videos/Ac_night/Ac_timelapse_night_low.mov
% ../Videos/Cb/Cb_timelapse_low.mov
% ../Videos/Ci_Cu/Ci_Cu_timelapse1_low.mov
% ../Videos/Ci_Cu/Ci_Cu_timelapse21_low.mp4
% ../Videos/Ci_Cu/Ci_Cu_timelapse22_low.mp4
% ../Videos/Cu/Cu_timelapse_Trim_low.mov
% ../Videos/Sc/sc_beneath_timelapse_150x_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov1_low.mov
% ../Videos/Ac_St_Cu/Ac_timelapseNov2_low.mov

[XSt1_train, XSt1_test, videoSt1] = importVideo('../Videos/St_fog/fog_video_above_timelapse_10x_low.mov',0.8);
[XSt2_train, XSt2_test, videoSt2] = importVideo('../Videos/St_fog/fog_video_near_60x_low.mov',0.8);
[XAc1_train, XAc1_test, videoAc1] = importVideo('../Videos/Ac_lenticularis/Ac_timelapse_sunrise_low.mov',0.8);
[XAc2_train, XAc2_test, videoAc2] = importVideo('../Videos/Ac_night/Ac_timelapse_night_low.mov',0.8);
[XCiCu1_train, XCiCu1_test, videoCiCu1] = importVideo('../Videos/Ci_Cu/Ci_Cu_timelapse1_low.mov',0.8);
[XCiCu2_train, XCiCu2_test, videoCiCu2] = importVideo('../Videos/Ci_Cu/Ci_Cu_timelapse21_low.mp4',0.8);
[XCiCu3_train, XCiCu3_test, videoCiCu3] = importVideo('../Videos/Ci_Cu/Ci_Cu_timelapse22_low.mp4',0.8);
[XCu1_train, XCu1_test, videoCu1] = importVideo('../Videos/Cu/Cu_timelapse_Trim_low.mov',0.8);
[XSc1_train, XSc1_test, videoSc1] = importVideo('../Videos/Sc/sc_beneath_timelapse_150x_low.mov',0.8);
[XAcStCu1_train, XAcStCu1_test, videoAcStCu1] = importVideo('../Videos/Ac_St_Cu/Ac_timelapseNov1_low.mov',0.8);
[XAcStCu2_train, XAcStCu2_test, videoAcStCu2] = importVideo('../Videos/Ac_St_Cu/Ac_timelapseNov2_low.mov',0.8);

%%

save ('VideoData.mat', '-v7.3');

%%
clear all
load ('VideoData.mat');

%%
load ('VideoData.mat', 'XCiCu1_train');