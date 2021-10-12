%% Create DMD data matrices
X1 = X(:, 1:end-1);
X2 = X(:, 2:end);

%% SVD and rank-2 truncation
r = 2; % rank truncation

%% calculate DMD
[Phi,omega,lambda,b,X_dmd,time_dynamics] = DMD(X1,X2,r,dt);
X_dmd(:,end+1) = X_dmd(:,end);           % to be able to display later
time_dynamics(:, end+1) = time_dynamics(:,end);

%% plot approximated function through DMD
subplot(2,2,4); 
surfl(real(X_dmd')); 
shading interp; colormap("copper"); view(-20,60);
set(gca, 'YTick', numel(t)/4 * (0:4)), 
set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
set(gca, 'XTick', linspace(1,numel(xi),3)), 
set(gca, 'Xticklabel',{'-10', '0', '10'});
title('f with DMD');
xlabel('x');
ylabel('t');
set(gca, 'FontSize', 14)
set(gcf, 'Color', 'w', 'Position', [400 400 600 400]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [10 10 16 12], 'PaperPositionMode', 'manual');
% print
% print('-djpeg', '-loose', ['figures/' sprintf('dmd_intro4_r=50.jpeg')]);

