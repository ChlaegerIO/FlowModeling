close all, clear all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\used_functions'

%% load video
% How many number of frame are used.
% nrOfFramesUsed = round(video.NumFrames/6);
nrOfFramesUsed = 130;

% read video
% ../Videos/St_fog/fog_video_above_timelapse_10x.mov
% ../Videos/St_fog/fog_video_above_timelapse.mov
% ../Videos/Ac/Ac_timelapse_sunrise.mp4
% ../Videos/Ac_night/Ac_timelapse_night.mp4
% ../Videos/Cb/Cb_timelapse.mov
% ../Videos/Ci_Cu/Ci_Cu_timelapse1.mp4
% ../Videos/Cu/Cu_timelapse_Trim.mp4
video = VideoReader('../Videos/Cu/Cu_timelapse_Trim_low.mov')
% save video within gray frames and make datamatrix X
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

X = X - min(X(:));
X = X ./max(X(:));
% X = X ./1.2;               % darken image, especially highlights

% print input video
% videoOut_input = VideoWriter('figures/St_slower_Video_Input','Grayscale AVI')
% open(videoOut_input);
% for i = 1:size(X,2)
%     frame_gray = reshape(X(:,i),nx,ny);
%     writeVideo(videoOut_input,frame_gray);
% end
% close(videoOut_input);

% free up space
clear frame, clear frame_gray;
X2 = X(:,2:end);
X = X(:,1:end-1);


%%  Compute DMD (Phi are eigenvectors)
[U,S,V] = svd(X,'econ');
r = 40;
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
% print('-djpeg', '-loose', ['figures/' sprintf('singularvalues_Cu_timelapse.jpeg')]);

%  Plot DMD spectrum
figure
set(gcf,'Position',[500 100 600 400])
theta = (0:1:100)*2*pi/100;
plot(cos(theta),sin(theta),'k--') % plot unit circle
hold on, grid on
scatter(real(diag(eigs)),imag(diag(eigs)),'ok')
axis([-1.1 1.1 -1.1 1.1]);
% print('-djpeg', '-loose', ['figures/' sprintf('eigenvalues_Cu_timelapse.jpeg')]);


%% plot first 24 POD modes
PODmodes = zeros(nx*2,ny*3);
count = 1;
U1 = U - min(U(:));
U1 = U1 ./max(U1(:));
U1 = tanh(15.*U1-9)./2 + 0.5;       % makes contrast higher of POD modes     
% for i=1:2
%     for j=1:3
%         if (i==1 && j==1)
%             PODmodes(1+(i-1)*nx:i*nx,1+(j-1)*ny:j*ny) = reshape(X(:,70),nx,ny);
%         else
%             PODmodes(1+(i-1)*nx:i*nx,1+(j-1)*ny:j*ny) = reshape(U1(:,4*(i-1)+20*j),nx,ny);
%         end
%         count = count + 1;
%     end
% end

figure, axes('position',[0  0  1  1]), axis off
imagesc(PODmodes), colormap gray
% print('-djpeg', '-loose', ['figures_v1/' sprintf('POD_Cu_timelapse_PODmodes_small.jpeg')]);


% free up space if necessary
sizeOfX = size(X,2);
clear lambda, 
%clear X2, clear x;
% clear X; 
% clear U

% could be improved with mrDMD!


%% prediction
factor = 2;                             % factor to advance time
dt = 1;                                 % tuning factor
until = factor*sizeOfX;
time_dynamics_pred = zeros(r, until);
t = (0:until-1)*dt;                     % time vector
for iter = 1:until
    time_dynamics_pred(:,iter) = (b.*exp(omega*t(iter)));
end
X_dmd_pred = Phi * time_dynamics_pred;

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
videoOut = VideoWriter('figures_v1/Cu_Video_prediction_out_factor2_r=40','Grayscale AVI')
open(videoOut);
for i = 1:size(X_dmd_pred,2)
    frame_gray_out = reshape(real(X_dmd_pred(:,i)),nx,ny);
    writeVideo(videoOut,frame_gray_out)
end
close(videoOut);

