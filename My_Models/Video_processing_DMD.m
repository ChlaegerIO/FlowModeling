close all, clear all, clc

%% load video
% How many number of frame are used.
% nrOfFramesUsed = round(video.NumFrames/6);
nrOfFramesUsed = 130;

% read video

% ../Videos/St_fog_real_timelapse/fog_video_above_timelapse_10x.mov
% ../Videos/Ac_lenticularis_timelapse_sunrise_short/Ac_timelapse_sunrise.mp4
% ../Videos/Ac_timelapse_night/Ac_timelapse_night.mp4
% ../Videos/Cb_timelapse/Cb_timelapse.mov
% ../Videos/Ci_Cu_timelapse/Ci_Cu_timelapse1.mp4
% ../Videos/Cu_timelapse/Cu_timelapse_Trim.mp4
% ../Videos/Sc_real_timelapse
video = VideoReader('../Videos/Cu_timelapse/Cu_timelapse_Trim.mp4')
% save video within gray frames and make datamatrix X
row = video.Height*video.Width;
X = zeros(row, nrOfFramesUsed);
ii = 1;
while hasFrame(video) && ii <= nrOfFramesUsed 
    frame = readFrame(video);
    frame_gray = double(rgb2gray(frame));
    X(:,ii) = reshape(frame_gray,[row, 1]);
    ii = ii + 1;
end

X = X - min(X(:));
X = X ./max(X(:));
X = X ./1.2;               % darken image, especially highlights

% print input video
% videoOut_input = VideoWriter('figures/Cu_Video_Input','Grayscale AVI')
% open(videoOut_input);
% for i = 1:size(X,2)
%     frame_gray = reshape(X(:,i),video.Height,video.Width);
%     writeVideo(videoOut_input,frame_gray)
% end
% close(videoOut_input);

% free up space
clear frame, clear frame_gray;
X2 = X(:,2:end);
X = X(:,1:end-1);

%%  Compute DMD (Phi are eigenvectors)
[U,S,V] = svd(X,'econ');

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
% print
% print('-djpeg', '-loose', ['figures/' sprintf('singularvalues_Cu_timelapse_Xin[0,1].jpeg')]);

r = 5;  % truncate at 21 modes, look at singular values
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);
Atilde = U'*X2*V*inv(S);
[W,eigs] = eig(Atilde);
Phi = X2*V*inv(S)*W;

lambda = diag(eigs); % discrete-time eigenvalues
omega = log(lambda); % continuous-time eigenvalues
x = X(:, 1);
b = Phi\x;

%  Plot DMD spectrum
figure
set(gcf,'Position',[500 100 600 400])
theta = (0:1:100)*2*pi/100;
plot(cos(theta),sin(theta),'k--') % plot unit circle
hold on, grid on
scatter(real(diag(eigs)),imag(diag(eigs)),'ok')
axis([-1.1 1.1 -1.1 1.1]);
% print('-djpeg', '-loose', ['figures/' sprintf('eigenvalues_Cu_timelapse_r=5.jpeg')]);

% free up space if necessary
sizeOfX = size(X,2);
clear lambda, clear U, clear X2, clear X, clear x;

%% prediction
factor = 2;                              % factor to advance time
until = factor*sizeOfX;
time_dynamics_pred = zeros(r, until);
t = (0:until-1); % time vector
for iter = 1:until
    time_dynamics_pred(:,iter) = (b.*exp(omega*t(iter)));
end
X_dmd_pred = Phi * time_dynamics_pred;

% recreate and make a prediction as a video
videoOut = VideoWriter('figures/Cu_Video_prediction_out_factor2_r=5','Grayscale AVI')
open(videoOut);
for i = 1:size(X_dmd_pred,2)
    frame_gray = reshape(real(X_dmd_pred(:,i)),video.Height,video.Width);
    writeVideo(videoOut,frame_gray)
end
close(videoOut);

