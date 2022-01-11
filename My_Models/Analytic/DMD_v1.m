close all, clear all, clc
addpath 'C:\Users\timok\Documents\Git_bachelor\FlowModeling\My_Models\Analytic\used_functions'

%% load video
% How many number of frame are used.
% nrOfFramesUsed = round(video.NumFrames/6);
nrOfFramesUsed = 200;

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
path_figure = 'figures_v1/Cu_2_'

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

X = matrixToNorm(X,0, 0.9);

% print input video
videoOut_input = VideoWriter(strcat(path_figure, 'Video_Input'),'Grayscale AVI')
open(videoOut_input);
for i = 1:size(X,2)
    frame_gray = reshape(X(:,i),nx,ny);
    writeVideo(videoOut_input,frame_gray);
end
close(videoOut_input);

% free up space
clear frame, clear frame_gray;
X2 = X(:,2:end);
X = X(:,1:end-1);


%%  Compute DMD (Phi are eigenvectors)
[U,S,V] = svd(X,'econ');
r = 190;
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
print('-djpeg', '-loose', [path_figure sprintf('singularvalues.jpeg')]);

%  Plot DMD spectrum
figure
set(gcf,'Position',[500 100 600 400])
theta = (0:1:100)*2*pi/100;
plot(cos(theta),sin(theta),'k--') % plot unit circle
hold on, grid on
scatter(real(diag(eigs)),imag(diag(eigs)),'ok')
axis([-1.1 1.1 -1.1 1.1]);
print('-djpeg', '-loose', [path_figure sprintf('eigenvalues.jpeg')]);


%% plot first 24 POD modes
PODmodes = zeros(nx*2,ny*3);
count = 1;  
for i=1:2
    for j=1:3
        if (i==1 && j==1)
            PODmodes(1+(i-1)*nx:i*nx,1+(j-1)*ny:j*ny) = reshape(X(:,70),nx,ny);
        else
            PODmodes(1+(i-1)*nx:i*nx,1+(j-1)*ny:j*ny) = matrixToNorm(reshape(U(:,30*(i-1)+3*j),nx,ny), 0, 0.9);
        end
        count = count + 1;
    end
end

figure, axes('position',[0  0  1  1]), axis off
imagesc(PODmodes), colormap gray
print('-djpeg', '-loose', [path_figure sprintf('PODmodes_small.jpeg')]);


% free up space if necessary
sizeOfX = size(X,2);
clear lambda, 
%clear X2, clear x;
% clear X; 
% clear U

% could be improved with mrDMD!


%% prediction
factor = 3;                             % factor to advance time
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
            if (X_dmd_pred(i,j)) < 0
                X_dmd_pred(i,j) = 0;
            elseif (X_dmd_pred(i,j)) > 1
                X_dmd_pred(i,j) = 1;
            end
        end
    end
end

% recreate and make a prediction as a video
videoOut = VideoWriter(strcat(path_figure, 'Video_predictionFactor3_r=190of200'),'Grayscale AVI')
open(videoOut);
for i = 1:size(X_dmd_pred,2)
    frame_gray_out = reshape(real(X_dmd_pred(:,i)),nx,ny);
    writeVideo(videoOut,frame_gray_out)
end
close(videoOut);

