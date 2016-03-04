%% CSCI567 Machine Learning HW2: Logistic Regression & Generative vs. Discriminative Model & Practical Logistic Regression
%
%  Instructions
%  ------------
%   
%   batchGD.m
%   batchGDL2.m
%   buildNeighborIdxs.m	
%   newtonMethod.m
%   normalize.m
%   plotBoarder.m
%   calculateAccuracy.m		
%   practicalLogistic.m
%   calculateCrossAccu.m	
%   randomShuffle.m
%   calculateGaussian.m		
%   separateLabel.m
%   calculate_pcc.m		
%   sigmoid.m
%   convertBin.m		
%   singleDiscretize.m
%   costFunction.m		
%   costFunctionPairWise.m	
%   costFunctionReg.m	
%   splitHalf.m
%   discretization.m		
%   discretize_real.m	
%   estimateMNL.m	
%   trainEvalModels.m
%   getHeldoutL2.m	
%   getTrTeAccuL2.m
%   train_model1.m
%   getRange.m	
%   train_model2.m
%   getSpecialHeldoutAccu.m	
%   train_model3.m
%   getTrTeAccu.m	
%   transferToMatrix.m
%  
%
%  @author Jiafeng Ni
%  USCID: 3144-3162-18
%

%% Initialization
clear ; close all; clc; 
% warning off;

fprintf('Running CSCI567_hw2_spring16... \n\n');
%% ==================== Part 1: Logistic Regression of Spam Email Detector using BatchGD ====================
%randomly shuffles the original data, split it into two halves, and
%separate features from labels
fprintf('Load data, randomly shuffle, split into two halves, and separate features from labels...\n');

originData = importdata('./spambase/spambase.data');
data = randomShuffle(originData);
[trainData testData] = splitHalf(data);

% separate features from labels
[origin_trainX trainY] = separateLabel(trainData);
[origin_testX testY] = separateLabel(testData);

% normalize origin data
normal_trainX = normalize(origin_trainX);
normal_testX = normalize(origin_testX);

% size of the training data
[trainM trainN] = size(origin_trainX);
[testM testN] = size(origin_testX);

% add bias, and initialize weights
bias_origin_trainX = [ones(trainM, 1) origin_trainX];
bias_origin_theta = zeros(trainN + 1, 1);
bias_normal_trainX = [ones(trainM, 1) normal_trainX];
bias_normal_theta = zeros(trainN + 1, 1);

bias_origin_testX = [ones(testM, 1) origin_testX];
bias_normal_testX = [ones(testM, 1) normal_testX];

fprintf('Data preprocess finished...\n\n');

fprintf('Part I: training data using batch Gradient Descent...\n');
%fprintf('Press enter to continue.\n');
%pause;

% choose step size and interation number
origin_alpha = 0.00002;
origin_iterations = 10000;
normal_alpha = 10;
normal_iterations = 200;

% train logistic regression using batch gradient descent on original data
fprintf('Train logistic regression using batch gradient descnet on original data set:\n');
fprintf('step size = %f, number of iterations = %d\n', origin_alpha, origin_iterations);
fprintf('please note that here default iteration number is %d, to gain higher accuracy,\n pelease try to change line %d to %d, but it will take a very line time to train!\n', 10000, 63, 1000000);
[origin_theta, J_origin, origin_accuracy] = batchGD(bias_origin_theta, bias_origin_trainX, trainY, origin_alpha, origin_iterations);
fprintf('Traing Accuracy  = %f\n', origin_accuracy(end));
% calculate test accuracies using original and normalized training models
origin_test_accu = calculateAccuracy(origin_theta, bias_origin_testX, testY);
fprintf('Testing Accuracy = %f\n\n', origin_test_accu);

fprintf('Train logistic regression using batch gradient descnet on normalized data set:\n');
fprintf('step size = %f, number of iterations = %d\n', normal_alpha, normal_iterations);
[normal_theta, J_normal, normal_accuracy] = batchGD(bias_normal_theta, bias_normal_trainX, trainY, normal_alpha, normal_iterations);
fprintf('Traing Accuracy  = %f\n', normal_accuracy(end));
% calculate test accuracies using original and normalized training models
normal_test_accu = calculateAccuracy(normal_theta, bias_normal_testX, testY);
fprintf('Testing Accuracy = %f\n\n', normal_test_accu);

% plot evolution of training accuracies as a function of training
% iterations
origin_iter = 1:origin_iterations;
normal_iter = 1:normal_iterations;
figure_batchGD_trainingAcc = figure();
figure(figure_batchGD_trainingAcc);
subplot(2,2,1), plot(origin_iter, origin_accuracy(origin_iter),'MarkerSize',8,'LineWidth',2);
hold on;
title('batchGD on original data','FontSize',20,'FontWeight','Demi');
xlabel('number of iterations','FontSize',20,'FontWeight','Demi');
ylabel('training accuracy','FontSize',20,'FontWeight','Demi');
subplot(2,2,2), plot(normal_iter, normal_accuracy,'MarkerSize',8,'LineWidth',2);
hold on;
title('batchGD on normalized data','FontSize',20,'FontWeight','Demi')
xlabel('number of iterations','FontSize',20,'FontWeight','Demi')
ylabel('training accuracy','FontSize',20,'FontWeight','Demi')

fprintf('Part I finished...\n\n');

%% ==================== Part 2: Logistic Regression of Spam Email Detector using Newton's method ====================
% using Newton's method to train the model

fprintf('Part II: training data using Newton method...\n');
%fprintf('Press enter to continue.\n');
%pause;

origin_newton_iterations = 15;
normal_newton_iterations = 15;

% train logistic regression using newton's method on original data
fprintf('Train logistic regression using Newton method on original data set:\n');
fprintf('number of iterations = %d\n', origin_newton_iterations);
[origin_newton_theta J_newton_origin origin_newton_accu] = newtonMethod(bias_origin_theta, bias_origin_trainX, trainY, origin_newton_iterations);
fprintf('Traing Accuracy  = %f\n', origin_newton_accu(end));
% calculate test accuracies using original and normalized training models
origin_newton_test_accu = calculateAccuracy(origin_newton_theta, bias_origin_testX, testY);
fprintf('Testing Accuracy = %f\n\n', origin_newton_test_accu);


% train logistic regression using newton's method on original data
fprintf('Train logistic regression using Newton method on normalized data set:\n');
fprintf('number of iterations = %d\n', normal_newton_iterations);
[normal_newton_theta J_newton_normal normal_newton_accu] = newtonMethod(bias_normal_theta, bias_normal_trainX, trainY, normal_newton_iterations);
fprintf('Traing Accuracy  = %f\n', normal_newton_accu(end));
% calculate test accuracies using original and normalized training models
normal_newton_test_accu = calculateAccuracy(normal_newton_theta, bias_normal_testX, testY);
fprintf('Testing Accuracy = %f\n\n', normal_newton_test_accu);

% plot evolution of training accuracies as a function of training
% iterations
origin_iter = 1:origin_newton_iterations;
normal_iter = 1:normal_newton_iterations;
subplot(2,2,3), plot(origin_iter, origin_newton_accu,'MarkerSize',8,'LineWidth',2);
hold on;
title('newton method on original data','FontSize',20,'FontWeight','Demi')
xlabel('number of iterations','FontSize',20,'FontWeight','Demi')
ylabel('training accuracy','FontSize',20,'FontWeight','Demi')
subplot(2,2,4), plot(normal_iter, normal_newton_accu,'MarkerSize',8,'LineWidth',2);
hold on;
title('newton method on normalized data','FontSize',20,'FontWeight','Demi')
xlabel('number of iterations','FontSize',20,'FontWeight','Demi')
ylabel('training accuracy','FontSize',20,'FontWeight','Demi')

fprintf('Part II finished...\n\n');

%% ==================== Part 3: Logistic Regression of Spam Email Detector using glmfit function ====================
% using glmfit function in MATLAB to train the model

fprintf('Part III: training data using glmfit function...\n');
%fprintf('Press enter to continue.\n');
%pause;

[glm_origin_theta origin_dev origin_stats] = glmfit(origin_trainX, trainY, 'binomial', 'link', 'logit');
glm_origin_accu = calculateAccuracy(glm_origin_theta, bias_origin_trainX, trainY);
glm_origin_test_accu = calculateAccuracy(glm_origin_theta, bias_origin_testX, testY);
fprintf('Train logistic regression using glmfit function on original data set:\n');
fprintf('Traing Accuracy  = %f\n', glm_origin_accu);
fprintf('Testing Accuracy = %f\n\n', glm_origin_test_accu);

[glm_normal_theta normal_dev normal_stats] = glmfit(normal_trainX, trainY, 'binomial', 'link', 'logit');
glm_normal_accu = calculateAccuracy(glm_normal_theta, bias_normal_trainX, trainY);
glm_normal_test_accu = calculateAccuracy(glm_normal_theta, bias_normal_testX, testY);
fprintf('Train logistic regression using glmfit function on normalized data set:\n');
fprintf('Traing Accuracy  = %f\n', glm_normal_accu);
fprintf('Testing Accuracy = %f\n\n', glm_normal_test_accu);

fprintf('Part III finished...\n\n');

fprintf('Accuracy Comparison:\n\n');
fprintf('Training:\n');
fprintf('    Accuracy              raw data              stadardized data\n');
fprintf('    batchGD               %f                  %f\n', origin_accuracy(end), normal_accuracy(end));
fprintf('    Newton Method         %f                  %f\n', origin_newton_accu(end), normal_newton_accu(end));
fprintf('    glmfit                %f                  %f\n\n', glm_origin_accu, glm_normal_accu);

fprintf('Accuracy Comparison:\n\n');
fprintf('Testing:\n');
fprintf('    Accuracy              raw data              stadardized data\n');
fprintf('    batchGD               %f                  %f\n', origin_test_accu, normal_test_accu);
fprintf('    Newton Method         %f                  %f\n', origin_newton_test_accu, normal_newton_test_accu);
fprintf('    glmfit                %f                  %f\n\n', glm_origin_test_accu, glm_normal_test_accu);


%% ==================== Part 4: Feature Analysis ====================
% discretize data, compute MI

fprintf('Part IV: Feature Analysis...\n');
%fprintf('Press enter to continue.\n');
%pause;

% calculate mi
mi = zeros(size(origin_trainX, 2), 1);

% for i = 1 : 55
%     mi(i) = discretize_real(origin_trainX(:, i), trainY);
% end
% 
% for i = 56 : 57
%     mi(i) = discretize_integer(origin_trainX(:, i), trainY);
% end
for i = 1 : 57
    mi(i) = discretize_real(origin_trainX(:, i), trainY);
end

% calculate pcc
pcc = calculate_pcc(origin_trainX, trainY);

% plot MI-vs-Features & PCC-vs-Features
feature_x = 1 : 57;
feature_x = feature_x';

fprintf('Plot MI and PCC vs Feature ID...\n\n');

MI_PCC_figure = figure();
figure(MI_PCC_figure);
subplot(1, 2, 1); plot(feature_x, mi, 'MarkerSize',8,'LineWidth',1);
title('MI vs FeatureID','FontSize',15,'FontWeight','Demi')
xlabel('FeatureID','FontSize',15,'FontWeight','Demi')
ylabel('MI','FontSize',15,'FontWeight','Demi')
hold on;
subplot(1, 2, 2); plot(feature_x, pcc, 'MarkerSize',8,'LineWidth',1);
title('PCC vs FeatureID','FontSize',15,'FontWeight','Demi')
xlabel('FeatureID','FontSize',15,'FontWeight','Demi')
ylabel('PCC','FontSize',15,'FontWeight','Demi')

[sorted_mi idx] = sort(mi, 'descend');
top_20 = idx(1:20);
fprintf('Feature ID with highest 20 MI selected: \n    ');
for i = 1 : 20
    fprintf('%d ',top_20(i));
end
fprintf('\n');

top_20_X = normal_trainX(:, top_20);
bias_top_20_X = [ones(trainM, 1) top_20_X];
[glm_20_theta glm_20_dev glm_20_stats] = glmfit(top_20_X, trainY, 'binomial', 'link', 'logit');
glm_20_accu = calculateAccuracy(glm_20_theta, bias_top_20_X, trainY);
glm_20_test_accu = calculateAccuracy(glm_20_theta, [ones(testM, 1), normal_testX(:, top_20)], testY);
fprintf('Train logistic regression using glmfit function on selected features of normalized data set:\n');
fprintf('Traing Accuracy  = %f\n', glm_20_accu);
fprintf('Testing Accuracy = %f\n\n', glm_20_test_accu);

top_20_pcc = pcc(top_20);
[sorted_top_20_pcc pcc_idx] = sort(top_20_pcc);
low_3_pcc_idx = pcc_idx(1:3);
other_17_pcc_idx = pcc_idx(4:end);
low_3_pcc = sorted_top_20_pcc(1:3);
fprintf('ID of selected features with lowest 3 PCC among the 20s: %d, %d, %d\n', low_3_pcc_idx(1), low_3_pcc_idx(2), low_3_pcc_idx(3));

%% 4.2.c discretize 3 features
other17_normX = top_20_X(:, other_17_pcc_idx);
these3_normX = top_20_X(:, low_3_pcc_idx);

normTestX = normal_testX(:, top_20);
normTestX17 = normTestX(:, other_17_pcc_idx);
normTestX3 = normTestX(:, low_3_pcc_idx);

togeX = [these3_normX; normTestX3];

fprintf('Discretize these 3 features into 10 equal width bins...\n');
normX1 = singleDiscretize(togeX(:,1), 10);
normX2 = singleDiscretize(togeX(:,2), 10);
normX3 = singleDiscretize(togeX(:,3), 10);

new_normX = [other17_normX normX1(1:trainM,:) normX2(1:trainM, :) normX3(1:trainM, :)];
bias_new_normX = [ones(trainM, 1) new_normX];
new_testX = [normTestX17 normX1((trainM+1):end,:) normX2((trainM+1):end,:) normX3((trainM+1):end,:)];
bias_new_testX = [ones(testM, 1) new_testX];
[discretized_theta discretized_dev discretized_stats] = glmfit(new_normX, trainY, 'binomial', 'link', 'logit');
discretized_train_accu = calculateAccuracy(discretized_theta, bias_new_normX, trainY);
discretized_test_accu = calculateAccuracy(discretized_theta, bias_new_testX, testY);

fprintf('Train logistic regression using glmfit function on 3 discretized features with other 17 normalized features:\n');
fprintf('Traing Accuracy  = %f\n', discretized_train_accu);
fprintf('Testing Accuracy = %f\n\n', discretized_test_accu);

fprintf('Part IV finished...\n\n');
fprintf('Problem 4.1 finished...\n\n');

fprintf('Problem 4.2 is ready to begin, please enter to continue...\n\n');
pause;


%% ==================== Part 5: Generative Model and Discriminative Model ====================
%problem 4.2
fprintf('Part V: Generative Model and Discriminative Model...\n');

trainEvalModels();

fprintf('Part V finished...\n\n');

fprintf('Problem 4.2 finished...\n\n');
fprintf('Problem 4.3 is ready to begin, please enter to continue...\n\n');
pause;


%% ==================== Part 6: Practical Logistic Regression on Toy Data ====================
% problem 4.3
fprintf('Part VI: Practical Logistic Regression on Toy Data...\n');

practicalLogistic();

fprintf('Part VI finished...\n\n');
fprintf('Problem 4.3 finished...\n\n');








