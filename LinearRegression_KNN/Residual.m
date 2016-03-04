function [r] = Residual(X, y, w)
% RESIDUAL computes the residual for linear regression

    r = y - X * w;
    
end