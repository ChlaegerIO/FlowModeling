%% PCA method
[U, S, V] = svd(X);
pc = zeros(length(U),r);
time_pc = zeros(length(V),r);
for i = 1:r
    pc(:,i) = U(:, i);          % store i-th PCA mode
    time_pc(:,i) = V(:, i);     % store i-th temporal evolution of pc
end

%% ICA method
[IC, ICt, ~] = fastica(real(X)');
% first ICA mode is in IC(1, :)
% first temporal evolution of ic1 is in ICt(:, 1)

%% ploting section
% variables for plots
f1_half = f1(:,201:400);
f2_half = f2(:,201:400);
X_dmd1 = Phi(:,1) * time_dynamics(1,:);
X_dmd1_half = X_dmd1(201:400,:);
X_dmd2 = Phi(:,2) * time_dynamics(2,:);
X_dmd2_half = X_dmd2(201:400,:);
f1_tempoMode = 1.6*sum(real(f1_half),2)/norm(real(f1_half));
dmd1_tempoMode = 1.6*transpose(sum(real(X_dmd2_half))/norm(real(X_dmd2_half)));
pc1_tempoMode = 10*real(time_pc(:,1));
ict1_tempoMode = 3*real(ICt(:,1));
f2_tempoMode = 1.3*sum(real(f2_half),2)/norm(real(f2_half));
dmd2_tempoMode = 1.3*transpose(sum(real(X_dmd1_half))/norm(real(X_dmd1_half)));
pc2_tempoMode = 10*real(time_pc(:,2));
ict2_tempoMode = 3*real(ICt(:,2));

% plot singular values
figure('Name', 'Singular values'), subplot(1,2,1)
semilogy(diag(S), 'k', 'LineWidth',2), grid on
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
% print('-djpeg', '-loose', ['figures/' sprintf('singular_values1.jpeg')]);

% plot Mode Error
figure('Name','Error comparision','position',[700 50 400 500]);
x = categorical({'Mode 1', 'Mode 2'});
y = [norm(f1_tempoMode - dmd1_tempoMode), norm(f1_tempoMode - pc1_tempoMode), norm(f1_tempoMode - ict1_tempoMode); 
    norm(f2_tempoMode - dmd2_tempoMode), norm(f2_tempoMode - pc2_tempoMode), norm(f2_tempoMode - ict2_tempoMode)];
barh(x,y)
xlabel('Error')
title('Error comparision')
legend({'DMD', 'PCA', 'ICA'})
set (gca, 'ydir', 'reverse' )
% print
% print('-djpeg', '-loose', ['figures/' sprintf('error1.jpeg')]);

% plot Temporal modes
figure('Name','Mode comparision with different methods','position',[500 50 1000 600]);
subplot(4,1,1); 
hold on
plot(T, f1_tempoMode,'b','LineWidth',[1.5]);
plot(T, dmd1_tempoMode,'r--','LineWidth',[1.5]);
plot(T, pc1_tempoMode,'g','LineWidth',[1.5]);
plot(T, ict1_tempoMode,'m','LineWidth',[1.5]);
set(gca, 'XTick', 0:pi:4*pi), 
set(gca, 'Xticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
title('Mode 1: Temporal');
xlabel('t');
hold off

subplot(4,1,2)
hold on
plot(T, f2_tempoMode,'b','LineWidth',[1.5]);
plot(T, dmd2_tempoMode,'r--','LineWidth',[1.5]);    %??
plot(T, pc2_tempoMode,'g','LineWidth',[1.5]);
plot(T, ict2_tempoMode,'m','LineWidth',[1.5]);
title('Mode 2: Temporal');
set(gca, 'XTick', 0:pi:4*pi), 
set(gca, 'Xticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
xlabel('t');
hold off

subplot(4,1,3); 
hold on
plo1 = plot(Xgrid, real(f1(1,:)),'b','LineWidth',[1.5]);
plo2 = plot(Xgrid, 6.5*real(Phi(:,2)'),'r--','LineWidth',[1.5]);
plo3 = plot(Xgrid, 6.5*real(pc(:,2)),'g','LineWidth',[1.5]);
plo4 = plot(Xgrid, 6.5*real(IC(2,:)/20),'m','LineWidth',[1.5]);       %what's with ic?
title('Mode 1: Spatial');
set(gca, 'XTick', -10:5:10), 
xlabel('x');
legend([plo1(1), plo2(1), plo3(1), plo4(1)],'True', 'DMD', 'PCA', 'ICA');
hold off

subplot(4,1,4)
hold on
plot(Xgrid, real(f2(1,:)),'b','LineWidth',[1.5]);
plot(Xgrid, -7.5*real(Phi(:,1)'),'r--','LineWidth',[1.5]);
plot(Xgrid, -7.5*real(pc(:,1)),'g','LineWidth',[1.5]);
plot(Xgrid, -7.5*real(IC(1,:)/20),'m','LineWidth',[1.5]);
title('Mode 2: Spatial');
set(gca, 'XTick', -10:5:10), 
xlabel('x');
hold off
set(gcf, 'Color', 'w', 'Position', [50 250 800 600]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [10 10 16 12], 'PaperPositionMode', 'manual');
% print
% print('-djpeg', '-loose', ['figures/' sprintf('dmd_pca_ica_comparision1.jpeg')]);

%% predicting the future
% size_dmd = size(X_dmd);
% nr_cols = size_dmd(2);
% cut_nr = 0;
% for cut = 1:nr_cols
%     % isn't equal enough?
%     if (X_dmd(10, cut) == X_dmd(10, nr_cols) && X_dmd(140, cut) == X_dmd(140, nr_cols))
%         cut_nr = cut;
%         break;
%     end
% end
% 
% X_dmd_tmp = X_dmd(:,cut_nr:nr_cols);
% X_dmd_fut = repmat(X_dmd_tmp,1,365);

