function [newtheta, J, accu] = newtonMethod(theta, X, y, iter)
%NEWTONMETHOD Compute cost and Hessian for logistic regression

% Initialize some useful values
[m n] = size(X); % number of training examples
J = 0; % cost 
accu = zeros(iter, 1); % training accuracy

for i= 1 : iter
   
    h = sigmoid(X*theta);
    %J = (-1 / m )*((y')*log(h) + (1-y)'*log(1 - h));
    grad = (1/m) * (X'*(h - y));
    H = zeros(n, n);
    
    for j = 1 : m
        sig = sigmoid(X(j, :) * theta);
        H = H + sig * (1 - sig) * (X(j, :))' * X(j, :);
    end
    H = H / m;
    theta = theta - pinv(H)*grad;
    h = round(sigmoid(X*theta));
    accu(i, 1) = length(find(h == y)) / m;
    
end

    newtheta = theta;


end