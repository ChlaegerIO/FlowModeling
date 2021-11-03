function makeVideo(pathToSave, X, nx, ny)
%MAKEVIDEO makes and saves the video in the specified path
%   pathToSave: path where video is saved
%   X: Video matrix with each frame in a column

videoOut_input = VideoWriter(pathToSave,'Grayscale AVI');
open(videoOut_input);
for i = 1:size(X,2)
    frame_gray = reshape(X(:,i),nx,ny);
    writeVideo(videoOut_input,frame_gray);
end
close(videoOut_input);

end

