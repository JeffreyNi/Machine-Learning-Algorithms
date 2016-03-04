function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
J = (-1 / m )*((y')*log(h) + (1-y)'*log(1 - h));
% J = J + (lambda / (2*m))*(sum(theta.^2)-theta(1)^2);
J = J + (lambda)*(sum(theta.^2)-theta(1)^2);

% grad = (1/m)*(X'*(h-y));
% n = length(theta);
% grad = grad + (lambda/m)*([0; theta(2:n, :)]);
grad = (X'*(h-y));
n = length(theta);
grad = grad + (lambda)*([0; theta(2:n, :)]);

% =============================================================

end