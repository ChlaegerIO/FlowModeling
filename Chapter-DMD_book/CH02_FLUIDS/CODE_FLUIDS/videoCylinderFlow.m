function videoCylinderFlow(toPlot,animation, nx, ny)
%VIDEOCYLINDERFLOW Make a video of the flow arround the cylinder
%   toPlot: data which will be plotted

figure;
load CCcool.mat 
colormap(CC);  % use custom colormap
vortmin = -5;  % only plot what is in -5 to 5 range
vortmax = 5;
toPlot(toPlot>vortmax) = vortmax;  % cutoff at vortmax
toPlot(toPlot<vortmin) = vortmin;  % cutoff at vortminaxis tight manual
% clean up axes
set(gca,'XTick',[1 50 100 150 200 250 300 350 400 449],'XTickLabel',{'-1','0','1','2','3','4','5','6','7','8'})
set(gca,'YTick',[1 50 100 150 199],'YTickLabel',{'2','1','0','-1','-2'});
set(gcf,'Position',[200 150 800 500])
axis equal
hold on

% Initialize video
myVideo = VideoWriter('figures/Cylinder_flow'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)

for i = 1:animation
    imagesc(reshape(toPlot(:,i),nx,ny)); % plot vorticity field
    % contour needs a lot of time --> slower video
%     contour(reshape(toPlot(:,i),nx,ny),[-5.5:.5:-.5 -.25 -.125],':k','LineWidth',1.2)
%     contour(reshape(toPlot(:,i),nx,ny),[.125 .25 .5:.5:5.5],'-k','LineWidth',1.2)
    theta = (1:100)/100'*2*pi;
    x = 49+25*sin(theta);
    y = 99+25*cos(theta);
    fill(x,y,[.3 .3 .3])  % place cylinder
    plot(x,y,'k','LineWidth',1.2) % cylinder boundary
    drawnow
%     pause(0.01) %Pause and grab frame
    F(i) = getframe(gcf)
    writeVideo(myVideo, F(i));
end

close(myVideo)

