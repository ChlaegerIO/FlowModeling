function [fig] = plotInOnePlot(fig_nr,title_str,fig_first, fig_last)
%UNTITLED Plot up to 12 plots in subplot(6,2, )
%
%   fig_nr: Number of the figure which is plotted
%   title_str: titel of figure
%   fig_from: starting figure to include
%   fig_to: last figure to include (the steps are 1)

figure(fig_nr)
set(gcf,'Position',[400 20 700 900])
title(title_str);
load CCcool.mat 
colormap(CC);
ax = zeros(12,1);
for i = 1:12
    ax(i)=subplot(6,2,i);
end
% Now copy contents of each figure over to destination figure
% Modify position of each axes as it is transferred
for i = fig_first:fig_last
    figure(i)
    h = get(gcf,'Children');
    newh = copyobj(h,fig_nr);
    posnewh = get(newh(1),'Position');
    possub  = get(ax(i),'Position');    
    if mod(i,2) == 0
        posnewh(1) = posnewh(1) + 0.2;
    else
        posnewh(1) = posnewh(1) - 0.2   ;
    end
    set(newh(1),'Position', [posnewh(1) possub(2) posnewh(3) possub(4)])
    delete(ax(i));
end
fig = figure(fig_nr)
