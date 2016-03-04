function [trainAccu testAccu] = getTrTeAccuL2(theta, x, y, xTe, yTe, lambdas)

num = length(lambdas);
trainAccu = [];
testAccu = [];
options = optimset('GradObj', 'on', 'display', 'off', 'MaxIter', 400);

for i = 1 : num
    [newTheta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, x, y, lambdas(i))), theta, options);
    
%     [newTheta, J, acc] = batchGDL2(theta, x, y, 1, 50, lambdas(i));
    accuTr = calculateAccuracy(newTheta, x, y);
    accuTe = calculateAccuracy(newTheta, xTe, yTe);
    trainAccu = [trainAccu accuTr];
    testAccu = [testAccu accuTe];
    
end


end