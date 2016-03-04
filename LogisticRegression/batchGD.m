function [newtheta, J, accu] = batchGD(theta, X, y, alpha, iter)
%BATCHGD Compute cost and gradient for logistic regression

% Initialize some useful values
m = length(y); % number of training examples
J = 0; % cost 
accu = zeros(iter, 1); % training accuracy

for i = 1 : iter
    
    h = sigmoid(X*theta);
    %J = (-1 / m )*((y')*log(h) + (1-y)'*log(1 - h));
    grad = (alpha/m) * (X'*(h - y));
    theta = theta - grad;

    h = round(sigmoid(X*theta));
    accu(i, :) = length(find(h == y)) / m;
    
end

newtheta = theta;

% =============================================================

end