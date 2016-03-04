%% CSCI567 Machine Learning HW1: Linear Regression && Classification

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%  normalEqn.m
%  ridgeNormalEqn.m
%  plotLRline.m
%  fitOutlier.m
%  plotLRlineWithLambda.m
%  LSC.m
%  Residual.m
%  decisionBoundary.m
%  decisionTree.m
%  knn_classify.m
%  normalize.m
%  plotSamples.m
%  preProcess.m
%  
%
%  @author Jiafeng Ni
%  USCID: 3144-3162-18
%

%% Initialization
clear ; close all; clc

fprintf('Running CSCI567_hw1_spring16 ... \n');
%% ==================== Part 1: Outlier Sample ====================
% Linear Regression using normal equation, outlier's influence on the
% training result
fprintf('First Part: Outlier Sample \n');
fprintf('Load data from "demoSynLR.mat" \n\n');

% Load Data
data = load('demoSynLR.mat');

% Train dataset1 and plot
fprintf('Program paused. \n    To train dataset1 using linear regression,\n    press enter to continue: \n');
pause;
data1 = data.data1;
X1 = data1.x;
y1 = data1.y;
figureLR = figure();
w1 = fitOutlier(figureLR, X1, y1, 1);

% Train dataset2 and plot
fprintf('Program paused. \n    To train dataset2 using linear regression,\n    press enter to continue: \n');
pause;
data2 = data.data2;
X2 = data2.x;
y2 = data2.y;
w2 = fitOutlier(figureLR, X2, y2, 2);

% Train dataset3 and plot
fprintf('Program paused. \n    To train dataset3 using linear regression,\n    press enter to continue: \n');
pause;
data3 = data.data3;
X3 = data3.x;
y3 = data3.y;
w3 = fitOutlier(figureLR, X3, y3, 3);

% Train dataset4 and plot
fprintf('Program paused. \n    To train dataset4 using linear regression,\n    press enter to continue: \n');
pause;
data4 = data.data4;
X4 = data4.x;
y4 = data4.y;
w4 = fitOutlier(figureLR, X4, y4, 4);

hold on;
fprintf('\n');
fprintf('Part1 Finished...\n');

%% ==================== Part 2: Weight Decay Coefficients ====================
% Implement normal equation using weight decay, compare influences of
% different weight decays on the result

fprintf('\n');
lambda = [0 0.1 1 10];

fprintf('Second Part: Weight Decay Coefficients \n');
fprintf('Program paused. \n    To train linear regression models for dataset1 \n    using weight decay coefficients lambda = {0.1,1,10} \n    respectively, press enter to continue: \n');
pause;
figureWD1 = figure(figure);
plotLRlineWithLambda(lambda, X1, y1, figureWD1);
hold on;

fprintf('Program paused. \n    To train linear regression models for dataset2 \n    using weight decay coefficients lambda = {0.1,1,10} \n    respectively, press enter to continue: \n');
pause;
figureWD2 = figure(figure);
plotLRlineWithLambda(lambda, X2, y2, figureWD2);
hold on;

fprintf('Program paused. \n    To train linear regression models for dataset3 \n    using weight decay coefficients lambda = {0.1,1,10} \n    respectively, press enter to continue: \n');
pause;
figureWD3 = figure(figure);
plotLRlineWithLambda(lambda, X3, y3, figureWD3);
hold on;

fprintf('Program paused. \n    To train linear regression models for dataset4 \n    using weight decay coefficients lambda = {0.1,1,10} \n    respectively, press enter to continue: \n');
pause;
figureWD4 = figure(figure);
plotLRlineWithLambda(lambda, X4, y4, figureWD4);
hold on;

fprintf('\n');
fprintf('Part2 Finished...\n');
fprintf('\n');

%% ==================== Part 3: Mark Outliers Using Different Methods ====================
% Calculate the 1. leverage score 2. studentized residual 3. Cook's
% distance for each data point, and pick out the outlier based on them

fprintf('Third Part: Recognize Extreme Samples \n');

fprintf('Program paused. \n    To get samples with largest {h, t, d} in dataset1\n    press enter to continue \n');
pause;
figureFindOutlier = figure(figure);
r1 = Residual(X1, y1, w1);
sigma1 = sqrt(var(r1));
[h1 t1 d1] = LSC(X1, r1, sigma1, 2);
plotSamples(figureFindOutlier,1, X1, y1, h1, t1, d1);

fprintf('Program paused. \n    To get samples with largest {h, t, d} in dataset2\n    press enter to continue \n');
pause;
r2 = Residual(X2, y2, w2);
sigma2 = sqrt(var(r2));
[h2 t2 d2] = LSC(X2, r2, sigma2, 2);
plotSamples(figureFindOutlier, 2, X2, y2, h2, t2, d2);

fprintf('Program paused. \n    To get samples with largest {h, t, d} in dataset3\n    press enter to continue \n');
pause;
r3 = Residual(X3, y3, w3);
sigma3 = sqrt(var(r3));
[h3 t3 d3] = LSC(X3, r3, sigma3, 2);
plotSamples(figureFindOutlier, 3, X3, y3, h3, t3, d3);

fprintf('Program paused. \n    To get samples with largest {h, t, d} in dataset4\n    press enter to continue \n');
pause;
r4 = Residual(X4, y4, w4);
sigma4 = sqrt(var(r4));
[h4 t4 d4] = LSC(X4, r4, sigma4, 2);
plotSamples(figureFindOutlier, 4, X4, y4, h4, t4, d4);

fprintf('\n');
fprintf('Part3 Finished...\n');
fprintf('\n');


%% ==================== Part 4: KNN classification ====================

fprintf('Forth Part: KNN classification \n    To train KNN using different K, and compare the influence of \n    different K, press enter to continue: \n');
pause;

% pre-process the input data
[trainX trainy] = preProcess('hw1_train.data');
[validX validy] = preProcess('hw1_validation.data');
[testX testy]   = preProcess('hw1_test.data');

% trainX = normalize(trainX);
% validX = normalize(validX);
% testX = normalize(testX);

for K = 1:2:15
   
    [valid_accu train_accu dist] = knn_classify(trainX, trainy, validX, validy, K);
    [test_accu train_accu dist] = knn_classify(trainX, trainy, testX, testy, K);
    
    fprintf('For K = %d, accuracies are: training: %f, validation: %f, test: %f \n' , K, train_accu, valid_accu, test_accu);
end
fprintf('\n');
fprintf('Part4 finished ...\n');

%% ==================== Part 5: Decision Tree ====================
fprintf('\n');
fprintf('Fifth Part: Decision Tree \n    To train decision tree using different number of \n    leaf nodes, and compare the accuracies, press enter to continue: \n');
pause;
fprintf('min# of leaf    train_accuracy_gini    train_accuracy_entropy    valid_accuracy_gini    valid_accuracy_entropy    test_accuracy_gini    test_accuracy_entopy\n');
for K = 1:10
   
    [new_accu_gini, train_accu_gini, new_accu_entro, train_accu_entro] = decisionTree(trainX, trainy, validX, validy, K);
    [test_accu_gini, train_accu_gini, test_accu_entro, train_accu_entro] = decisionTree(trainX, trainy, testX, testy, K);
    if K ~= 10
        fprintf('     %d                %f                 %f                %f                %f                %f                %f\n', K, train_accu_gini, train_accu_entro, new_accu_gini, new_accu_entro, test_accu_gini, test_accu_entro);
    else 
        fprintf('    %d                %f                 %f                %f                %f                %f                %f\n', K, train_accu_gini, train_accu_entro, new_accu_gini, new_accu_entro, test_accu_gini, test_accu_entro);  
    end
end

fprintf('\n');
fprintf('Part5 finished ...\n');
fprintf('\n');

%% ==================== Part 6: Decision Boundary ====================

fprintf('Part6 Decision Boundary\n    To plot decision boundary using KNN under different\n    K values, please enter to continue:\n');
pause;

boundaryData = load('hw1boundary.mat');
knn_data = boundaryData.features;
knn_label = boundaryData.labels;

decisionBoundary(knn_data, knn_label, 1);
decisionBoundary(knn_data, knn_label, 5);
decisionBoundary(knn_data, knn_label, 15);
[X y] = decisionBoundary(knn_data, knn_label, 25);

fprintf('Part6 finished ...\n');
fprintf('\n');
fprintf('All parts finished ...\n exit...\n');


