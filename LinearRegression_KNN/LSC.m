function [h t d] = LSC(X, r, sigma, k)
% LSC compute leverage score, studentized residual, and Cooker's distance
    X = X(:, 2);
    H = X * pinv(X' * X) * X';
    m = size(H, 1);
    h = zeros(1, m);
    t = zeros(1, m);
    d = zeros(1, m);
    
    for i = 1:1:m
        h(i) = H(i, i);
        t(i) = r(i) / (sigma * sqrt(1 - h(i)));
        d(i) = (h(i) / (1 - h(i))) * ((t(i))^2 / (1 + k));
    end
    
end