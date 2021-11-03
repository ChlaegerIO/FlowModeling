function [U,S,V] = rsvd(X,r,q,p);
% rsvd: randomized SVD with power iterations (q) and oversampling (p)
%   X: data matrix
%   r: rank
%   q: how many power iterations
%   p: oversampling r+p

% sample column space of X with P
ny = size(X,2);
P = randn(ny,r+p);
Z = X*P;
for k=1:q
    Z = X*(X'*Z);
end

[Q,R] = qr(Z,0);

% compute SVD on projected Y=Q'*X space
Y = Q'*X;
[UY,S,V] = svd(Y,'econ');
U = Q*UY;

end