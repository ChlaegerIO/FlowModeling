close all, clear all, clc

%% load video
video = VideoReader('../Videos/Cu_timelapse/Cu_timelapse_Trim.mp4')
% all_frames = zeros(Video.NumFrames,1);
ii = 1;
row = video.Height*video.Width;
X = zeros(row, round(video.NumFrames/6));
while hasFrame(video) && ii <= round(video.NumFrames/6) 
    frame = readFrame(video);
    frame_gray = double(rgb2gray(frame));
    X(:,ii) = reshape(frame_gray,[row, 1]);
    ii = ii + 1;
end

clear frame, clear frame_gray;
X2 = X(:,2:end);
X = X(:,1:end-1);

%%  Compute DMD (Phi are eigenvectors)
[U,S,V] = svd(X,'econ');

% plot singular values
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
print('-djpeg', '-loose', ['figures/' sprintf('singular_values_Cu_timelapse.jpeg')]);

r = 100;  % truncate at 21 modes
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


%% prediction