function trainEvalModels()
    load toyGMM.mat
    
    %% MLE learning of model1, Gaussian Discriminative Analysis I
    % your code here
    % e.g. 
    % model1 = ...;
    % acc1 = ...;
    fprintf('MLE learning of model 1, Gaussian Discriminative Analysis I.\n');
    x1_train = dataTr.x1;
    x1_test = dataTe.x1;
    x2_train = dataTr.x2;
    x2_test = dataTe.x2;
    x3_train = dataTr.x3;
    x3_test = dataTe.x3;
    x_train = [x1_train; x2_train; x3_train];   
    x_test = [x1_test; x2_test; x3_test];
    
    size1 = size(x1_train, 1);
    size2 = size(x2_train, 1);
    size3 = size(x3_train, 1);
    ttl_size = size1 + size2 + size3;
    y1_train = ones(size1, 1);
    y2_train = ones(size2, 1) * 2;
    y3_train = ones(size3, 1) * 3;
    y_train = [y1_train; y2_train; y3_train];
    
    size1_test = size(x1_test, 1);
    size2_test = size(x2_test, 1);
    size3_test = size(x3_test, 1);
    ttl_size_test = size1_test + size2_test + size3_test;
    y1_test = ones(size1_test, 1);
    y2_test = ones(size2_test, 1) * 2;
    y3_test = ones(size3_test, 1) * 3;
    y_test = [y1_test; y2_test; y3_test];
    
    pi1 = size1 / ttl_size;
    pi2 = size2 / ttl_size;
    pi3 = size3 / ttl_size;
    mu1 = mean(x1_train);
    mu2 = mean(x2_train);
    mu3 = mean(x3_train);
    S1 = sqrt(var(x1_train));
    S2 = sqrt(var(x2_train));
    S3 = sqrt(var(x3_train));
%     S1 = cov(x1_train);
%     S2 = cov(x2_train);
%     S3 = cov(x3_train);
    fprintf('parameters for model1:\n');
    fprintf('----pi1 = %f\n', pi1);
    fprintf('----pi2 = %f\n', pi2);
    fprintf('----pi3 = %f\n', pi3);
    fprintf('----mu1 = %f, %f\n', mu1(1), mu1(2));
    fprintf('----mu2 = %f, %f\n', mu2(1), mu2(2));
    fprintf('----mu3 = %f, %f\n', mu3(1), mu3(2));
    fprintf('---- S1 = %f,       0\n', S1(1));
    fprintf('              0   ,    %f\n', S1(2));
    fprintf('---- S2 = %f,       0\n', S2(1));
    fprintf('              0   ,    %f\n', S2(2));
    fprintf('---- S3 = %f,       0\n', S3(1));
    fprintf('              0   ,    %f\n', S3(2));
    
    model1.pi = [pi1 pi2 pi3];
    model1.m1 = mu1;
    model1.m2 = mu2;
    model1.m3 = mu3;
    model1.S1 = diag(S1);
    model1.S2 = diag(S2);
    model1.S3 = diag(S3);
%     model1.S1 = S1;
%     model1.S2 = S2;
%     model1.S3 = S3;
    
%     y_model1 = zeros(ttl_size, 1);
%     
%     for i = 1 : ttl_size
%         
%         p1 = calculateGaussian(x_train(i, :), mu1, S1) * pi1;
%         p2 = calculateGaussian(x_train(i, :), mu2, S2) * pi2;
%         p3 = calculateGaussian(x_train(i, :), mu3, S3) * pi3;
%         p = [p1 p2 p3];
%         [val y_model1(i)] = max(p);
%         
%     end
%     
%     model1_train_accu = length(find(y_model1 == y_train)) / ttl_size;
%     %fprintf('Training accuracy for model1: %f\n', model1_train_accu);
%     
%     y_test_model1 = zeros(ttl_size_test, 1);
%     
%     for i = 1 : ttl_size_test
%         
%        p1 = calculateGaussian(x_test(i, :), mu1, S1) * pi1;
%        p2 = calculateGaussian(x_test(i, :), mu2, S2) * pi2;
%        p3 = calculateGaussian(x_test(i, :), mu3, S3) * pi3;
%        p = [p1 p2 p3];
%        [val y_test_model1(i)] = max(p);
%         
%     end
%     
%     model1_test_accu = length(find(y_test_model1 == y_test)) / ttl_size_test;
    [model1_train_accu, model1_test_accu] = train_model1(x1_train, x2_train, x3_train, x1_test, x2_test, x3_test);
    fprintf('Training accuracy for model1: %f\n', model1_train_accu);
    fprintf('Testing accuracy for model1: %f\n\n', model1_test_accu);
    
    %% MLE learning of model2, Gaussian Discriminative Analysis II
    % your code here
    fprintf('MLE learning of model 2, Gaussian Discriminative Analysis II.\n');
    S = sqrt(var(x_train));
    S1 = S;
    S2 = S;
    S3 = S;
    fprintf('parameters for model2:\n');
    fprintf('----pi1 = %f\n', pi1);
    fprintf('----pi2 = %f\n', pi2);
    fprintf('----pi3 = %f\n', pi3);
    fprintf('----mu1 = %f, %f\n', mu1(1), mu1(2));
    fprintf('----mu2 = %f, %f\n', mu2(1), mu2(2));
    fprintf('----mu3 = %f, %f\n', mu3(1), mu3(2));
    fprintf('---- S1 = %f,       0\n', S1(1));
    fprintf('              0   ,    %f\n', S1(2));
    fprintf('---- S2 = %f,       0\n', S2(1));
    fprintf('              0   ,    %f\n', S2(2));
    fprintf('---- S3 = %f,       0\n', S3(1));
    fprintf('              0   ,    %f\n', S3(2));
    
    model2.pi = [pi1 pi2 pi3];
    model2.m1 = mu1;
    model2.m2 = mu2;
    model2.m3 = mu3;
    model2.S1 = diag(S1);
    model2.S2 = diag(S2);
    model2.S3 = diag(S3);
    
%     y_model2 = zeros(ttl_size, 1);
%     
%     for i = 1 : ttl_size
%         
%         p1 = calculateGaussian(x_train(i, :), mu1, S1) * pi1;
%         p2 = calculateGaussian(x_train(i, :), mu2, S2) * pi2;
%         p3 = calculateGaussian(x_train(i, :), mu3, S3) * pi3;
%         p = [p1 p2 p3];
%         [val y_model2(i)] = max(p);
%         
%     end
%     
%     model2_train_accu = length(find(y_model2 == y_train)) / ttl_size;
%     fprintf('Training accuracy for model2: %f\n', model2_train_accu);
%     
%     y_test_model2 = zeros(ttl_size_test, 1);
%     
%     for i = 1 : ttl_size_test
%         
%        p1 = calculateGaussian(x_test(i, :), mu1, S1) * pi1;
%        p2 = calculateGaussian(x_test(i, :), mu2, S2) * pi2;
%        p3 = calculateGaussian(x_test(i, :), mu3, S3) * pi3;
%        p = [p1 p2 p3];
%        [val y_test_model2(i)] = max(p);
%     
%     end
%     
%     model2_test_accu = length(find(y_test_model2 == y_test)) / ttl_size_test;

    [model2_train_accu, model2_test_accu] = train_model2(x1_train, x2_train, x3_train, x1_test, x2_test, x3_test);
    fprintf('Training accuracy for model2: %f\n', model2_train_accu);
    fprintf('Testing accuracy for model2: %f\n\n', model2_test_accu);
    
    %% learning of model3, the MLR classifeir
    % your code here
    fprintf('Learning of model3, the MLR classifier.\n');
    B = mnrfit(x_train, y_train);
    fprintf('parameters for model3:\n');
    beta1 = B(:, 1)';
    beta2 = B(:, 2)';
    beta3 = [0 0 0];
    fprintf('----beta1 = %f, %f, %f\n', beta1(1), beta1(2), beta1(3));
    fprintf('----beta2 = %f, %f, %f\n', beta2(1), beta2(2), beta2(3));
    fprintf('----beta3 = %f, %f, %f\n\n', beta3(1), beta3(2), beta3(3));
    
    model3.w = [beta1(2:3) beta1(1); beta2(2:3) beta2(1); 0 0 0];
            
%     y_model3 = estimateMNL(x_train, B);
%     
%     model3_train_accu = length(find(y_model3 == y_train)) / ttl_size;
%     fprintf('Training accuracy for model3: %f\n', model3_train_accu);
%     
%     y_test_model3 = estimateMNL(x_test, B);
%     
%     model3_test_accu = length(find(y_test_model3 == y_test)) / ttl_size_test;
%     fprintf('Testing accuracy for model3: %f\n\n', model3_test_accu);

    [model3_train_accu, model3_test_accu] = train_model3(x1_train, x2_train, x3_train, x1_test, x2_test, x3_test);
    fprintf('Training accuracy for model3: %f\n', model3_train_accu);
    fprintf('Testing accuracy for model3: %f\n\n', model3_test_accu);
    
    %% visualize and compare learned models
    % plotBoarder(model1, model2, model3, dataTe)
    fprintf('visualize and compare three learned models:\n');

    plotBoarder(model1, model2, model3, dataTe);
    
    fprintf('visualization finished...\n\n');
    %% randomly select {1%, 5%, 10%, 25%, 50%, 100%} of training data. Train 3 models.
    fprintf('Train three models using randomly-picked training data:\n');
    
    ratios = [1 5 10 25 50 100];
    test_accus = zeros(6, 3);
        
    for i = 1 : length(ratios)
        
       fprintf('%d. Randomly pick %d%% of training data:\n', i, ratios(i));
       perm1 = randperm(size1);
       perm2 = randperm(size2);
       perm3 = randperm(size3);
       sz1 = ratios(i) * size1 / 100;
       sz2 = ratios(i) * size2 / 100;
       sz3 = ratios(i) * size3 / 100;
       
       rg1 = perm1(1:sz1);
       rg2 = perm2(1:sz2);
       rg3 = perm3(1:sz3);
       
       x1_tr = x1_train(rg1, :);
       x2_tr = x2_train(rg2, :);
       x3_tr = x3_train(rg3, :);

       [train_accu1, test_accu1] = train_model1(x1_tr, x2_tr, x3_tr, x1_test, x2_test, x3_test);
       [train_accu2, test_accu2] = train_model2(x1_tr, x2_tr, x3_tr, x1_test, x2_test, x3_test);
       [train_accu3, test_accu3] = train_model3(x1_tr, x2_tr, x3_tr, x1_test, x2_test, x3_test);
       test_accus(i, 1) = test_accu1;
       test_accus(i, 2) = test_accu2;
       test_accus(i, 3) = test_accu3;
        
       fprintf('    model1: training accuracy = %f, testing accuracy = %f\n', train_accu1, test_accu1);
       fprintf('    model2: training accuracy = %f, testing accuracy = %f\n', train_accu2, test_accu2);
       fprintf('    model3: training accuracy = %f, testing accuracy = %f\n', train_accu3, test_accu3);
    end
    
    accu_vs_datasize = figure();
    figure(accu_vs_datasize);
    subplot(1,3,1), plot(ratios' ./ 100, test_accus(:, 1), 'MarkerSize',8,'LineWidth',1);
    axis([0 1 0.5 1]);
    hold on;
    title('model1','FontSize', 15,'FontWeight','Demi');
    xlabel('data size percentage','FontSize',15,'FontWeight','Demi');
    ylabel('testing accuracy','FontSize',15,'FontWeight','Demi');
    subplot(1,3,2), plot(ratios' ./ 100, test_accus(:, 2), 'MarkerSize',8,'LineWidth',1);
    axis([0 1 0.5 1]);
    hold on;
    title('model2','FontSize', 15,'FontWeight','Demi');
    xlabel('data size percentage','FontSize',15,'FontWeight','Demi');
    ylabel('testing accuracy','FontSize',15,'FontWeight','Demi');
    subplot(1,3,3), plot(ratios' ./ 100, test_accus(:, 3), 'MarkerSize',8,'LineWidth',1);
    axis([0 1 0.5 1]);
    hold on;
    title('model3','FontSize', 15,'FontWeight','Demi');
    xlabel('data size percentage','FontSize',15,'FontWeight','Demi');
    ylabel('testing accuracy','FontSize',15,'FontWeight','Demi');
    
    
