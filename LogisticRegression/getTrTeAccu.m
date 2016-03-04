function [train_accu test_accu] = getTrTeAccu(x, y, xTe, yTe)

    theta = glmfit(x, y, 'binomial', 'link', 'logit');
    train_accu = calculateAccuracy(theta, [ones(size(x,1), 1) x], y);
    test_accu = calculateAccuracy(theta, [ones(size(xTe,1), 1) xTe], yTe);

end