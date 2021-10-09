%% Define time and space discretizations
xi = linspace(-10,10,400);
t = linspace(0,4*pi,200); 
dt = t(2) - t(1);
[Xgrid,T] = meshgrid(xi,t);

%% Create two spatio-temporal patterns
f1 = sech(Xgrid+3).*(1*exp(1j*2.3*T));
f2 = (sech(Xgrid).*tanh(Xgrid)).*(2*exp(1j*2.8*T));
f3 = 0.5.*(sech(Xgrid + 5)).*(exp(1j*2.3*T)).*(tanh(T-pi)-tanh(T-3*pi));
%f3 = sech(Xgrid + 6 - T).*exp(1j*2.3*T);

%% Combine signals and make data matrix
plt1Use = 0;
plt2Use = 1;
plt3Use = 1;
f = f2 + f3;
X = f.'; % Data Matrix

%% Visualize fi, fj, and f
figure;
if plt1Use == 1
    subplot(2,2,1); 
    surfl(real(f1)); 
    shading interp; colormap("copper"); view(-20,60);
    set(gca, 'YTick', numel(t)/4 * (0:4)), 
    set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
    set(gca, 'XTick', linspace(1,numel(xi),3)), 
    set(gca, 'Xticklabel',{'-10', '0', '10'});
    title('f1');
    xlabel('x');
    ylabel('t');

    if plt2Use == 1
        subplot(2,2,2);
        surfl(real(f2));
        shading interp; colormap("copper"); view(-20,60);
        set(gca, 'YTick', numel(t)/4 * (0:4)), 
        set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
        set(gca, 'XTick', linspace(1,numel(xi),3)), 
        set(gca, 'Xticklabel',{'-10', '0', '10'});
        title('f2');
        xlabel('x');
        ylabel('t');

        subplot(2,2,3);
        surfl(real(f)); 
        shading interp; colormap("copper"); view(-20,60);
        set(gca, 'YTick', numel(t)/4 * (0:4)), 
        set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
        set(gca, 'XTick', linspace(1,numel(xi),3)), 
        set(gca, 'Xticklabel',{'-10', '0', '10'});
        title('f = f1 + f2');
        xlabel('x');
        ylabel('t');
    else
        subplot(2,2,2);
        surfl(real(f3));
        shading interp; colormap("copper"); view(-20,60);
        set(gca, 'YTick', numel(t)/4 * (0:4)), 
        set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
        set(gca, 'XTick', linspace(1,numel(xi),3)), 
        set(gca, 'Xticklabel',{'-10', '0', '10'});
        title('f3');
        xlabel('x');
        ylabel('t');

        subplot(2,2,3);
        surfl(real(f)); 
        shading interp; colormap("copper"); view(-20,60);
        set(gca, 'YTick', numel(t)/4 * (0:4)), 
        set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
        set(gca, 'XTick', linspace(1,numel(xi),3)), 
        set(gca, 'Xticklabel',{'-10', '0', '10'});
        title('f = f1 + f3');
        xlabel('x');
        ylabel('t');
    end
else
    subplot(2,2,1);
    surfl(real(f2));
    shading interp; colormap("copper"); view(-20,60);
    set(gca, 'YTick', numel(t)/4 * (0:4)), 
    set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
    set(gca, 'XTick', linspace(1,numel(xi),3)), 
    set(gca, 'Xticklabel',{'-10', '0', '10'});
    title('f2');
    xlabel('x');
    ylabel('t');

    subplot(2,2,2);
    surfl(real(f3));
    shading interp; colormap("copper"); view(-20,60);
    set(gca, 'YTick', numel(t)/4 * (0:4)), 
    set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
    set(gca, 'XTick', linspace(1,numel(xi),3)), 
    set(gca, 'Xticklabel',{'-10', '0', '10'});
    title('f3');
    xlabel('x');
    ylabel('t');

    subplot(2,2,3);
    surfl(real(f)); 
    shading interp; colormap("copper"); view(-20,60);
    set(gca, 'YTick', numel(t)/4 * (0:4)), 
    set(gca, 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
    set(gca, 'XTick', linspace(1,numel(xi),3)), 
    set(gca, 'Xticklabel',{'-10', '0', '10'});
    title('f = f2 + f3');
    xlabel('x');
    ylabel('t');
end
