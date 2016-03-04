function [J] = costFunctionPairWise(theta, X, y, lambda, S)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

h = sigmoid(X*theta);
J = (-1 / m )*((y')*log(h) + (1-y)'*log(1 - h));

n = size(S, 1);
panish = 0;
for i = 1 : n
   
    panish = panish + (theta(S(i, 1)) - theta(S(i, 2)))^2;
    
end

J = J + lambda * panish / 2;


% =============================================================

end