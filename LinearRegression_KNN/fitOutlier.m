function [w] = fitOutlier(figureLR, X, y, subID)
% train a linear regression model with and without outlier sample. Then
% plot the data points and fitted lines with each subplot.

w = normalEqn(X, y);
w_no_out = normalEqn(X(1 : end - 1, :), y(1 : end - 1));
plotLRline(figureLR, subID, w, w_no_out, X, y);

end