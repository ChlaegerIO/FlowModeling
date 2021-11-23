function [X_out] = matrixToNorm(X, offset, factor)
%MatrixToNorm: special normalization of the real part to [0,factor]
%   X: Input matrix
%   factor: factor to scale after transformation, if factor > 1 then
%   Interval is [0,factor]

X_out = X - min(real(X(:))) + offset;
X_out = X_out ./max(real(X_out(:)));
X_out = X_out .*factor;                % darken image, especially highlights
end