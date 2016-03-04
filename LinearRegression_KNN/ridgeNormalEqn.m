function [theta] = normalEqn(X, y, lambda)
%RIDGENORMALEQN Computes the closed-form solution to linear regression 
%   RIDGENORMALEQN(X,y, lambda) computes the closed-form solution to linear 
%   regression using the normal equations, considering the weight decay.

theta = zeros(size(X, 2), 1);
n = size(X, 2);
diag = lambda * eye(n);
theta = pinv(ctranspose(X)*X + diag)*(X'*y);

% ============================================================

end
