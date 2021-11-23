clear all; close all; clc

load catData_w.mat; load dogData_w.mat; CD=[dog_wave cat_wave];
train=[dog_wave(:,1:60) cat_wave(:,1:60)];
test=[dog_wave(:,61:80) cat_wave(:,61:80)];
label=[ones(60,1); -1*ones(60,1)].';
test_real_labels=[ones(20,1); -1*ones(20,1)].';

A=label*pinv(train); test_labels=sign(A*test);
subplot(5,1,1), bar(test_real_labels)
subplot(5,1,2), bar(test_labels) 
subplot(5,1,3), bar(A)
figure(2), subplot(2,2,1)
A2=flipud(reshape(A,32,32)); pcolor(A2), colormap(gray)

figure(1), subplot(5,1,4)
A=lasso(train.',label.','Lambda',0.1).'; 
test_labels=sign(A*test);
bar(test_labels)
subplot(5,1,5)
bar(A)
figure(2), subplot(2,2,2)
A2=flipud(reshape(A,32,32)); pcolor(A2), colormap(gray)