function [X_train, X_test, video] = importVideo(path,trainPercent)
%importVideo imports a video
%   path: path where the video is found
%   trainPercent: % of frames used for training

video = VideoReader(path)
nrOfFramesUsed = round(video.NumFrames);
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

X_train = X(:,1:round(trainPercent*nrOfFramesUsed)-1);
X_test = X(:, round(trainPercent*nrOfFramesUsed):nrOfFramesUsed);

end