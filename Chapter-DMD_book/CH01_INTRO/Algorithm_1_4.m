%% PCA method
[U, S, V] = svd(X);
pc1 = U(:, 1); % first PCA mode
pc2 = U(:, 2); % second PCA mode
time_pc1 = V(:, 1); % temporal evolution of pc1
time_pc2 = V(:, 2); % temporal evolution of pc2

%% ICA method
[IC, ICt, ~] = fastica(real(X)');
ic1 = IC(1, :); % first ICA mode
ic2 = IC(2, :); % second ICA mode
time_ic1 = ICt(:, 1); % temporal evolution of ic1
time_ic2 = ICt(:, 2); % temporal evolution of ic2

%% ploting section
% plot singular values
figure('Name', 'Singular values');
S_values = diag(S);
S_values_norm = S_values/sum(S_values);
plot(S_values_norm(1:20),'o');
xlabel('k');
ylabel('\sigma_k')

% plot Mode Error
%figure('Name', 'Mode Error');
%plot(norm(f-Phi(:,1))/norm(f), abs(f-pc1), abs(f-ic1), abs(f-Phi(:,2)), abs(f-pc2), abs(f-ic2));

% plot Temporal modes
figure('Name','Mode comparision with different methods','position',[50 50 1000 600]);
subplot(4,1,1); 
hold on
plot(T, real(f2(:,140)),'b','LineWidth',[1.5]);
plot(T, real(X_dmd(140,:)'),'r--','LineWidth',[1.5]);
plot(T, real(time_pc1),'g','LineWidth',[1.5]);
plot(T, real(time_ic1),'m','LineWidth',[1.5]);
set(gca, 'XTick', 0:pi:4*pi), 
set(gca, 'Xticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
title('Mode 1: Temporal');
xlabel('t');
hold off

subplot(4,1,2)
hold on
plot(T, real(f3(:,220)),'b','LineWidth',[1.5]);
plot(T, real(X_dmd(220,:)'),'r--','LineWidth',[1.5]);
plot(T, real(time_pc2),'g','LineWidth',[1.5]);
plo4 = plot(T, real(time_ic2),'m','LineWidth',[1.5]);
title('Mode 2: Temporal');
set(gca, 'XTick', 0:pi:4*pi), 
set(gca, 'Xticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
xlabel('t');
hold off

% plot spatial modes
for i = 1:400
    f2_mod_norm(i) = f2(9,i)/max(max(f2));          % which cross-section?
end

subplot(4,1,3); 
hold on
plo1 = plot(Xgrid, real(f2_mod_norm),'b','LineWidth',[1.5]);
plo2 = plot(Xgrid, real(Phi(:,1)'),'r--','LineWidth',[1.5]);
plo3 = plot(Xgrid, real(pc1),'g','LineWidth',[1.5]);
plo4 = plot(Xgrid,real(ic1/20),'m','LineWidth',[1.5]);       %what's with ic?
title('Mode 1: Spatial');
set(gca, 'XTick', -10:5:10), 
xlabel('x');
legend([plo1(1), plo2(1), plo3(1), plo4(1)],'True', 'DMD', 'PCA', 'ICA');
hold off

subplot(4,1,4)
hold on
plot(Xgrid, real(f3(53,:)),'b','LineWidth',[1.5]);
plot(Xgrid, real(Phi(:,2)'),'r--','LineWidth',[1.5]);
plot(Xgrid, real(pc2),'g','LineWidth',[1.5]);
plot(Xgrid, real(ic2/20),'m','LineWidth',[1.5]);
title('Mode 2: Spatial');
set(gca, 'XTick', -10:5:10), 
xlabel('x');
hold off

set(gcf, 'Color', 'w', 'Position', [200 200 800 600]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [10 10 16 12], 'PaperPositionMode', 'manual');
print('-djpeg', '-loose', ['figures/' sprintf('dmd_pca_ica_comparision4.jpeg')]);

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

