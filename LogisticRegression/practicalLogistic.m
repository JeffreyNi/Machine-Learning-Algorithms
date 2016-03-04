function practicalLogistic()
%     warning off;
    data = load('toySpiral.mat');
    data1 = data.data1;
    data2 = data.data2;
    data3 = data.data3;
    data4 = data.data4;
    
    grid_size = [2 4 8 16];
    range = [-2 2];
    
    % initializing all y's
    y1 = data1.yTr - 1;
    y1Te = data1.yTe - 1;
    y2 = data2.yTr -1;
    y2Te = data2.yTe - 1;
    y3 = data3.yTr - 1;
    y3Te = data3.yTe - 1;
    y4 = data4.yTr - 1;
    y4Te = data4.yTe - 1;
    
    x1Tr = data1.xTr;
    x2Tr = data2.xTr;
    x3Tr = data3.xTr;
    x4Tr = data4.xTr;
    
    %% choosing the optimal size for dataset1
    cross_accu1 = zeros(11, 4);

    for i = 1 : 4
        sz = grid_size(i);
        x1 = discretization(x1Tr, sz, range);
        cross_accu1(:, i) = calculateCrossAccu(x1, y1);
    end
    
    [cross_accu1_chosen grid_chosen1_idx] = max(cross_accu1(11, :));
    grid_size_chosen1 = grid_size(grid_chosen1_idx);

    x1Tr = discretization(data1.xTr, grid_size_chosen1, range);
    x1Te = discretization(data1.xTe, grid_size_chosen1, range);
    
    
    %% choosing the optimal size for dataset2        
    cross_accu2 = zeros(11, 4);
    
    for i = 1 : 4
        sz = grid_size(i);
        x2 = discretization(x2Tr, sz, range);
        cross_accu2(:, i) = calculateCrossAccu(x2, y2);
    end
    
    [cross_accu2_chosen grid_chosen2_idx] = max(cross_accu2(11, :));
    grid_size_chosen2 = grid_size(grid_chosen2_idx);
    
    x2Tr = discretization(data2.xTr, grid_size_chosen2, range);
    x2Te = discretization(data2.xTe, grid_size_chosen2, range);
    %% choosing the optimal size for dataset3        
    cross_accu3 = zeros(11, 4);
    
    for i = 1 : 4
        sz = grid_size(i);
        x3 = discretization(x3Tr, sz, range);
        cross_accu3(:, i) = calculateCrossAccu(x3, y3);
    end
    
    [cross_accu3_chosen grid_chosen3_idx] = max(cross_accu3(11, :));
    grid_size_chosen3 = grid_size(grid_chosen3_idx);
    
    x3Tr = discretization(data3.xTr, grid_size_chosen3, range);
    x3Te = discretization(data3.xTe, grid_size_chosen3, range);
    %% choosing the optimal size for dataset4        
    cross_accu4 = zeros(11, 4);
    
    for i = 1 : 4
        sz = grid_size(i);
        x4 = discretization(x4Tr, sz, range);
        cross_accu4(:, i) = calculateCrossAccu(x4, y4);
    end

    [cross_accu4_chosen grid_chosen4_idx] = max(cross_accu4(11, :));
    grid_size_chosen4 = grid_size(grid_chosen4_idx);

    x4Tr = discretization(data4.xTr, grid_size_chosen4, range);
    x4Te = discretization(data4.xTe, grid_size_chosen4, range);
    %% print all accuracies for dataset1
    fprintf('Choosing optimal grid size for dataset1:\n');
    fprintf('Cross Validation Accuracies For Different grid sizes:\n\n');
    fprintf('    accuracy         2              4              8             16\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f       %f       %f       %f\n', i, cross_accu1(i, 1), cross_accu1(i, 2), cross_accu1(i, 3), cross_accu1(i, 4));
        else 
            fprintf('      No.%d      %f       %f       %f       %f\n', i, cross_accu1(i, 1), cross_accu1(i, 2), cross_accu1(i, 3), cross_accu1(i, 4));
        end
    end
    fprintf('      Mean       %f       %f       %f       %f\n\n', cross_accu1(11, 1), cross_accu1(11, 2), cross_accu1(11, 3), cross_accu1(11, 4));
    fprintf('Chosen grid size for dataset1: %d\n\n', grid_size_chosen1);
    
    %% print all accuracies for dataset2
    fprintf('Choosing optimal grid size for dataset2:\n');
    fprintf('Cross Validation Accuracies For Different grid sizes:\n\n');
    fprintf('    accuracy         2              4              8             16\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f       %f       %f       %f\n', i, cross_accu2(i, 1), cross_accu2(i, 2), cross_accu2(i, 3), cross_accu2(i, 4));
        else 
            fprintf('      No.%d      %f       %f       %f       %f\n', i, cross_accu2(i, 1), cross_accu2(i, 2), cross_accu2(i, 3), cross_accu2(i, 4));
        end
    end
    fprintf('      Mean       %f       %f       %f       %f\n\n', cross_accu2(11, 1), cross_accu2(11, 2), cross_accu2(11, 3), cross_accu2(11, 4));
    fprintf('Chosen grid size for dataset2: %d\n\n', grid_size_chosen2);
    
    %% print all accuracies for dataset3
    fprintf('Choosing optimal grid size for dataset3:\n');
    fprintf('Cross Validation Accuracies For Different grid sizes:\n\n');
    fprintf('    accuracy         2              4              8             16\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f       %f       %f       %f\n', i, cross_accu3(i, 1), cross_accu3(i, 2), cross_accu3(i, 3), cross_accu3(i, 4));
        else 
            fprintf('      No.%d      %f       %f       %f       %f\n', i, cross_accu3(i, 1), cross_accu3(i, 2), cross_accu3(i, 3), cross_accu3(i, 4));
        end
    end
    fprintf('      Mean       %f       %f       %f       %f\n\n', cross_accu3(11, 1), cross_accu3(11, 2), cross_accu3(11, 3), cross_accu3(11, 4));
    fprintf('Chosen grid size for dataset3: %d\n\n', grid_size_chosen3);
    
    %% print all accuracies for dataset4
    fprintf('Choosing optimal grid size for dataset4:\n');
    fprintf('Cross Validation Accuracies For Different grid sizes:\n\n');
    fprintf('    accuracy         2              4              8             16\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f       %f       %f       %f\n', i, cross_accu4(i, 1), cross_accu4(i, 2), cross_accu4(i, 3), cross_accu4(i, 4));
        else 
            fprintf('      No.%d      %f       %f       %f       %f\n', i, cross_accu4(i, 1), cross_accu4(i, 2), cross_accu4(i, 3), cross_accu4(i, 4));
        end
    end
    fprintf('      Mean       %f       %f       %f       %f\n\n', cross_accu4(11, 1), cross_accu4(11, 2), cross_accu4(11, 3), cross_accu4(11, 4));
    fprintf('Chosen grid size for dataset4: %d\n\n', grid_size_chosen4);
    
    %% calculate training and testing accuracies
    [train_accu1 test_accu1] = getTrTeAccu(x1Tr, y1, x1Te, y1Te);
    [train_accu2 test_accu2] = getTrTeAccu(x2Tr, y2, x2Te, y2Te);
    [train_accu3 test_accu3] = getTrTeAccu(x3Tr, y3, x3Te, y3Te);
    [train_accu4 test_accu4] = getTrTeAccu(x4Tr, y4, x4Te, y4Te);
    
    fprintf('Discretization Summarize:\n\n');
    fprintf('    Statistics     train_accu     heldout_accu     test_accu      chosen_grid\n');
    fprintf('    dataset 1       %f        %f        %f           %d\n', train_accu1, cross_accu1_chosen, test_accu1, grid_size_chosen1);
    fprintf('    dataset 2       %f        %f        %f           %d\n', train_accu2, cross_accu2_chosen, test_accu2, grid_size_chosen2);
    fprintf('    dataset 3       %f        %f        %f           %d\n', train_accu3, cross_accu3_chosen, test_accu3, grid_size_chosen3);
    fprintf('    dataset 4       %f        %f        %f           %d\n\n', train_accu4, cross_accu4_chosen, test_accu4, grid_size_chosen4);
    fprintf('====================================================================================\n\n');
    
    %% l2 normalization
    x1Trl2 = discretization(data1.xTr, 8, range);
    x2Trl2 = discretization(data2.xTr, 8, range);
    x3Trl2 = discretization(data3.xTr, 8, range);
    x4Trl2 = discretization(data4.xTr, 8, range);
    randPerm1 = randperm(size(x1Trl2, 1));
    randPerm2 = randperm(size(x2Trl2, 1));
    randPerm3 = randperm(size(x3Trl2, 1));
    randPerm4 = randperm(size(x4Trl2, 1));
    x1Trl2 = x1Trl2(randPerm1, :);
    x2Trl2 = x2Trl2(randPerm2, :);
    x3Trl2 = x3Trl2(randPerm3, :);
    x4Trl2 = x4Trl2(randPerm4, :);
    y1 = y1(randPerm1);
    y2 = y2(randPerm2);
    y3 = y3(randPerm3);
    y4 = y4(randPerm4);
    
    x1Tel2 = discretization(data1.xTe, 8, range);
    x2Tel2 = discretization(data2.xTe, 8, range);
    x3Tel2 = discretization(data3.xTe, 8, range);
    x4Tel2 = discretization(data4.xTe, 8, range);
    
    lambdas = [1 0.1 0.01 0.001];
    
    x1Trl2_bias = [ones(size(x1Trl2, 1), 1) x1Trl2];
    x2Trl2_bias = [ones(size(x2Trl2, 1), 1) x2Trl2];
    x3Trl2_bias = [ones(size(x3Trl2, 1), 1) x3Trl2];
    x4Trl2_bias = [ones(size(x4Trl2, 1), 1) x4Trl2];
    
    x1Tel2_bias = [ones(size(x1Tel2, 1), 1) x1Tel2];
    x2Tel2_bias = [ones(size(x2Tel2, 1), 1) x2Tel2];
    x3Tel2_bias = [ones(size(x3Tel2, 1), 1) x3Tel2];
    x4Tel2_bias = [ones(size(x4Tel2, 1), 1) x4Tel2];
    
    initTheta1 = zeros(size(x1Trl2_bias, 2), 1);
    initTheta2 = zeros(size(x2Trl2_bias, 2), 1);
    initTheta3 = zeros(size(x3Trl2_bias, 2), 1);
    initTheta4 = zeros(size(x4Trl2_bias, 2), 1);
    
    heldout_Accu1_l2 = zeros(11, 4);
    for i = 1 : 4
        heldout_Accu1_l2(:, i) = getHeldoutL2(x1Trl2_bias, y1, lambdas(i));
    end
    
    [max_healdout_Accu1 idx1] = max(heldout_Accu1_l2(11, :));
    lambda_chosen1 = lambdas(idx1);
    
    heldout_Accu2_l2 = zeros(11, 4);
    for i = 1 : 4
        heldout_Accu2_l2(:, i) = getHeldoutL2(x2Trl2_bias, y2, lambdas(i));
    end
    
    [max_healdout_Accu2 idx2] = max(heldout_Accu2_l2(11, :));
    lambda_chosen2 = lambdas(idx2);
    
    heldout_Accu3_l2 = zeros(11, 4);
    for i = 1 : 4
        heldout_Accu3_l2(:, i) = getHeldoutL2(x3Trl2_bias, y3, lambdas(i));
    end
    
    [max_healdout_Accu3 idx3] = max(heldout_Accu3_l2(11, :));
    lambda_chosen3 = lambdas(idx3);
    
    heldout_Accu4_l2 = zeros(11, 4);
    for i = 1 : 4
        heldout_Accu4_l2(:, i) = getHeldoutL2(x4Trl2_bias, y4, lambdas(i));
    end
    
    [max_healdout_Accu4 idx4] = max(heldout_Accu4_l2(11, :));
    lambda_chosen4 = lambdas(idx4);
    
    [trainAccu1_L2 testAccu1_L2] = getTrTeAccuL2(initTheta1, x1Trl2_bias, y1, x1Tel2_bias, y1Te, lambdas);
    [trainAccu2_L2 testAccu2_L2] = getTrTeAccuL2(initTheta2, x2Trl2_bias, y2, x2Tel2_bias, y2Te, lambdas);
    [trainAccu3_L2 testAccu3_L2] = getTrTeAccuL2(initTheta3, x3Trl2_bias, y3, x3Tel2_bias, y3Te, lambdas);
    [trainAccu4_L2 testAccu4_L2] = getTrTeAccuL2(initTheta4, x4Trl2_bias, y4, x4Tel2_bias, y4Te, lambdas);

    %% print all accuracies for dataset1
    fprintf('Choosing optimal coefficient for dataset1:\n');
    fprintf('Cross Validation Accuracies For Different lambdas:\n\n');
    fprintf('    accuracy         1              0.1              0.01             0.001\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f        %f         %f          %f\n', i, heldout_Accu1_l2(i, 1), heldout_Accu1_l2(i, 2), heldout_Accu1_l2(i, 3), heldout_Accu1_l2(i, 4));
        else 
            fprintf('      No.%d      %f        %f         %f          %f\n', i, heldout_Accu1_l2(i, 1), heldout_Accu1_l2(i, 2), heldout_Accu1_l2(i, 3), heldout_Accu1_l2(i, 4));
        end
    end
    fprintf('      Mean       %f        %f         %f          %f\n\n', heldout_Accu1_l2(11, 1), heldout_Accu1_l2(11, 2), heldout_Accu1_l2(11, 3), heldout_Accu1_l2(11, 4));
    fprintf('Chosen coefficient for dataset1: %f\n\n', lambda_chosen1);
    
    
    %% print all accuracies for dataset2
    fprintf('Choosing optimal coefficient for dataset2:\n');
    fprintf('Cross Validation Accuracies For Different lambdas:\n\n');
    fprintf('    accuracy         1              0.1              0.01             0.001\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f        %f         %f          %f\n', i, heldout_Accu2_l2(i, 1), heldout_Accu2_l2(i, 2), heldout_Accu2_l2(i, 3), heldout_Accu2_l2(i, 4));
        else 
            fprintf('      No.%d      %f        %f         %f          %f\n', i, heldout_Accu2_l2(i, 1), heldout_Accu2_l2(i, 2), heldout_Accu2_l2(i, 3), heldout_Accu2_l2(i, 4));
        end
    end
    fprintf('      Mean       %f        %f         %f          %f\n\n', heldout_Accu2_l2(11, 1), heldout_Accu2_l2(11, 2), heldout_Accu2_l2(11, 3), heldout_Accu2_l2(11, 4));
    fprintf('Chosen coefficient for dataset2: %f\n\n', lambda_chosen2);

    %% print all accuracies for dataset3
    fprintf('Choosing optimal coefficient for dataset3:\n');
    fprintf('Cross Validation Accuracies For Different lambdas:\n\n');
    fprintf('    accuracy         1              0.1              0.01             0.001\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f        %f         %f          %f\n', i, heldout_Accu3_l2(i, 1), heldout_Accu3_l2(i, 2), heldout_Accu3_l2(i, 3), heldout_Accu3_l2(i, 4));
        else 
            fprintf('      No.%d      %f        %f         %f          %f\n', i, heldout_Accu3_l2(i, 1), heldout_Accu3_l2(i, 2), heldout_Accu3_l2(i, 3), heldout_Accu3_l2(i, 4));
        end
    end
    fprintf('      Mean       %f        %f         %f          %f\n\n', heldout_Accu3_l2(11, 1), heldout_Accu3_l2(11, 2), heldout_Accu3_l2(11, 3), heldout_Accu3_l2(11, 4));
    fprintf('Chosen coefficient for dataset3: %f\n\n', lambda_chosen3);
    
    %% print all accuracies for dataset4
    fprintf('Choosing optimal coefficient for dataset4:\n');
    fprintf('Cross Validation Accuracies For Different lambdas:\n\n');
    fprintf('    accuracy         1              0.1              0.01             0.001\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f        %f         %f          %f\n', i, heldout_Accu4_l2(i, 1), heldout_Accu4_l2(i, 2), heldout_Accu4_l2(i, 3), heldout_Accu4_l2(i, 4));
        else 
            fprintf('      No.%d      %f        %f         %f          %f\n', i, heldout_Accu4_l2(i, 1), heldout_Accu4_l2(i, 2), heldout_Accu4_l2(i, 3), heldout_Accu4_l2(i, 4));
        end
    end
    fprintf('      Mean       %f        %f         %f          %f\n\n', heldout_Accu4_l2(11, 1), heldout_Accu4_l2(11, 2), heldout_Accu4_l2(11, 3), heldout_Accu4_l2(11, 4));
    fprintf('Chosen coefficient for dataset4: %f\n\n', lambda_chosen4);
    
    
    fprintf('L2 normalize accuracies for dataset1:\n\n');
    fprintf('    accuracy             1              0.1            0.01             0.001\n');
    fprintf('    train_accu       %f        %f        %f         %f\n', trainAccu1_L2(1), trainAccu1_L2(2), trainAccu1_L2(3), trainAccu1_L2(4));
    fprintf('    heldout_accu     %f        %f        %f         %f\n', heldout_Accu1_l2(11,1), heldout_Accu1_l2(11,2), heldout_Accu1_l2(11,3), heldout_Accu1_l2(11,4));
    fprintf('    test_accu        %f        %f        %f         %f\n\n', testAccu1_L2(1), testAccu1_L2(2), testAccu1_L2(3), testAccu1_L2(4));
    
    fprintf('L2 normalize accuracies for dataset2:\n\n');
    fprintf('    accuracy             1              0.1            0.01             0.001\n');
    fprintf('    train_accu       %f        %f        %f         %f\n', trainAccu2_L2(1), trainAccu2_L2(2), trainAccu2_L2(3), trainAccu2_L2(4));
    fprintf('    heldout_accu     %f        %f        %f         %f\n', heldout_Accu2_l2(11,1), heldout_Accu2_l2(11,2), heldout_Accu2_l2(11,3), heldout_Accu2_l2(11,4));
    fprintf('    test_accu        %f        %f        %f         %f\n\n', testAccu2_L2(1), testAccu2_L2(2), testAccu2_L2(3), testAccu2_L2(4));
    
    fprintf('L2 normalize accuracies for dataset3:\n\n');
    fprintf('    accuracy             1              0.1            0.01             0.001\n');
    fprintf('    train_accu       %f        %f        %f         %f\n', trainAccu3_L2(1), trainAccu3_L2(2), trainAccu3_L2(3), trainAccu3_L2(4));
    fprintf('    heldout_accu     %f        %f        %f         %f\n', heldout_Accu3_l2(11,1), heldout_Accu3_l2(11,2), heldout_Accu3_l2(11,3), heldout_Accu3_l2(11,4));
    fprintf('    test_accu        %f        %f        %f         %f\n\n', testAccu3_L2(1), testAccu3_L2(2), testAccu3_L2(3), testAccu3_L2(4));
    
    fprintf('L2 normalize accuracies for dataset4:\n\n');
    fprintf('    accuracy             1              0.1            0.01             0.001\n');
    fprintf('    train_accu       %f        %f        %f         %f\n', trainAccu4_L2(1), trainAccu4_L2(2), trainAccu4_L2(3), trainAccu4_L2(4));
    fprintf('    heldout_accu     %f        %f        %f         %f\n', heldout_Accu4_l2(11,1), heldout_Accu4_l2(11,2), heldout_Accu4_l2(11,3), heldout_Accu4_l2(11,4));
    fprintf('    test_accu        %f        %f        %f         %f\n\n', testAccu4_L2(1), testAccu4_L2(2), testAccu4_L2(3), testAccu4_L2(4));
    fprintf('====================================================================================\n\n');
    
    %% plotting 3D-bar for dataset2 & dataset3
    fprintf('Plotting 3D-bar for dataset2 & dataset3...\n\n');
    
    options = optimset('GradObj', 'on', 'display', 'off', 'MaxIter', 400);
    theta2_unnorm = glmfit(x2Trl2, y2, 'binomial', 'link', 'logit');
    [theta2_norm J2 ac2] = fminunc(@(t)(costFunctionReg(t, x2Trl2_bias, y2, lambda_chosen2)), initTheta2, options);
    theta3_unnorm = glmfit(x3Trl2, y3, 'binomial', 'link', 'logit');
    [theta3_norm J3 ac3] = fminunc(@(t)(costFunctionReg(t, x3Trl2_bias, y3, lambda_chosen3)), initTheta3, options);
    
    
    theta2_unnorm = transferToMatrix(theta2_unnorm);
    theta2_norm = transferToMatrix(theta2_norm);
    theta3_unnorm = transferToMatrix(theta3_unnorm);
    theta3_norm = transferToMatrix(theta3_norm);
    
    fig_theta2_unnorm = figure(6);
    figure(fig_theta2_unnorm);
    bar3(theta2_unnorm);
    title('unregularized dataset2','FontSize',15,'FontWeight','Demi');
    fig_theta2_norm = figure(7);
    figure(fig_theta2_norm);
    bar3(theta2_norm);
    title('regularized dataset2','FontSize',15,'FontWeight','Demi');
    fig_theta3_unnorm = figure(8);
    figure(fig_theta3_unnorm);
    bar3(theta3_unnorm);
    title('unregularized dataset3','FontSize',15,'FontWeight','Demi');
    fig_theta3_norm = figure(9);
    figure(fig_theta3_norm);
    bar3(theta3_norm);
    title('regularized dataset3','FontSize',15,'FontWeight','Demi');
    
    fprintf('====================================================================================\n\n');

    %% Special Regularization
    S = buildNeighborIdxs(8);
    S = S + 1;
    options = optimset('display', 'off', 'MaxIter', 400);
    
    [theta1_batch J1_batch accuracy1] = fminunc(@(t)(costFunction(t, x1Trl2_bias, y1)), initTheta1, options);
    [theta2_batch J2_batch accuracy2] = fminunc(@(t)(costFunction(t, x2Trl2_bias, y2)), initTheta2, options);
    [theta3_batch J3_batch accuracy3] = fminunc(@(t)(costFunction(t, x3Trl2_bias, y3)), initTheta3, options);
    [theta4_batch J4_batch accuracy4] = fminunc(@(t)(costFunction(t, x4Trl2_bias, y4)), initTheta4, options);
    
    % dataset1
    heldout_special1_accu = zeros(11, 4);
    for i = 1 : 4
        heldout_special1_accu(:, i) = getSpecialHeldoutAccu(x1Trl2_bias, y1, lambdas(i), S);
    end
    
    [max_healdout_special_Accu1 idx1] = max(heldout_special1_accu(11, :));
    special_lambda_chosen1 = lambdas(idx1);
    special_theta1 = fminunc(@(t)(costFunctionPairWise(t, x1Trl2_bias, y1, special_lambda_chosen1, S)), initTheta1, options);
    special_train_accu1 = calculateAccuracy(special_theta1, x1Trl2_bias, y1);
    special_test_accu1 = calculateAccuracy(special_theta1, x1Tel2_bias, y1Te);
    
    % dataset2
    heldout_special2_accu = zeros(11, 4);
    for i = 1 : 4
        heldout_special2_accu(:, i) = getSpecialHeldoutAccu(x2Trl2_bias, y2, lambdas(i), S);
    end
    
    [max_healdout_special_Accu2 idx2] = max(heldout_special2_accu(11, :));
    special_lambda_chosen2 = lambdas(idx2);
    special_theta2 = fminunc(@(t)(costFunctionPairWise(t, x2Trl2_bias, y2, special_lambda_chosen2, S)), initTheta2, options);
    special_train_accu2 = calculateAccuracy(special_theta2, x2Trl2_bias, y2);
    special_test_accu2 = calculateAccuracy(special_theta2, x2Tel2_bias, y2Te);
    
    % dataset3
    heldout_special3_accu = zeros(11, 4);
    for i = 1 : 4
        heldout_special3_accu(:, i) = getSpecialHeldoutAccu(x3Trl2_bias, y3, lambdas(i), S);
    end
    
    [max_healdout_special_Accu3 idx3] = max(heldout_special3_accu(11, :));
    special_lambda_chosen3 = lambdas(idx3);
    special_theta3 = fminunc(@(t)(costFunctionPairWise(t, x3Trl2_bias, y3, special_lambda_chosen3, S)), initTheta3, options);
    special_train_accu3 = calculateAccuracy(special_theta3, x3Trl2_bias, y3);
    special_test_accu3 = calculateAccuracy(special_theta3, x3Tel2_bias, y3Te);
    
    heldout_special4_accu = zeros(11, 4);
    for i = 1 : 4
        heldout_special4_accu(:, i) = getSpecialHeldoutAccu(x4Trl2_bias, y4, lambdas(i), S);
    end
    
    [max_healdout_special_Accu4 idx4] = max(heldout_special4_accu(11, :));
    special_lambda_chosen4 = lambdas(idx4);
    special_theta4 = fminunc(@(t)(costFunctionPairWise(t, x4Trl2_bias, y4, special_lambda_chosen4, S)), initTheta4, options);
    special_train_accu4 = calculateAccuracy(special_theta4, x4Trl2_bias, y4);
    special_test_accu4 = calculateAccuracy(special_theta4, x4Tel2_bias, y4Te);
    
    
    %% print all accuracies for dataset1
    fprintf('Special Logistic Regression with pairwise regularization\n\n');
    fprintf('Choosing optimal coefficient for dataset1:\n');
    fprintf('Cross Validation Accuracies For Different grid sizes:\n\n');
    fprintf('    accuracy         1              0.1              0.01             0.001\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f        %f         %f          %f\n', i, heldout_special1_accu(i, 1), heldout_special1_accu(i, 2), heldout_special1_accu(i, 3), heldout_special1_accu(i, 4));
        else 
            fprintf('      No.%d      %f        %f         %f          %f\n', i, heldout_special1_accu(i, 1), heldout_special1_accu(i, 2), heldout_special1_accu(i, 3), heldout_special1_accu(i, 4));
        end
    end
    fprintf('      Mean       %f        %f         %f          %f\n\n', heldout_special1_accu(11, 1), heldout_special1_accu(11, 2), heldout_special1_accu(11, 3), heldout_special1_accu(11, 4));
    fprintf('Chosen coefficient for dataset1: %f\n\n', special_lambda_chosen1);
    
    
    %% print all accuracies for dataset2
    fprintf('Choosing optimal coefficient for dataset2:\n');
    fprintf('Cross Validation Accuracies For Different grid sizes:\n\n');
    fprintf('    accuracy         1              0.1              0.01             0.001\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f        %f         %f          %f\n', i, heldout_special2_accu(i, 1), heldout_special2_accu(i, 2), heldout_special2_accu(i, 3), heldout_special2_accu(i, 4));
        else 
            fprintf('      No.%d      %f        %f         %f          %f\n', i, heldout_special2_accu(i, 1), heldout_special2_accu(i, 2),heldout_special2_accu(i, 3), heldout_special2_accu(i, 4));
        end
    end
    fprintf('      Mean       %f        %f         %f          %f\n\n', heldout_special2_accu(11, 1), heldout_special2_accu(11, 2), heldout_special2_accu(11, 3), heldout_special2_accu(11, 4));
    fprintf('Chosen coefficient for dataset2: %f\n\n', special_lambda_chosen2);

    %% print all accuracies for dataset3
    fprintf('Choosing optimal coefficient for dataset3:\n');
    fprintf('Cross Validation Accuracies For Different grid sizes:\n\n');
    fprintf('    accuracy         1              0.1              0.01             0.001\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f        %f         %f          %f\n', i, heldout_special3_accu(i, 1), heldout_special3_accu(i, 2), heldout_special3_accu(i, 3), heldout_special3_accu(i, 4));
        else 
            fprintf('      No.%d      %f        %f         %f          %f\n', i, heldout_special3_accu(i, 1), heldout_special3_accu(i, 2), heldout_special3_accu(i, 3), heldout_special3_accu(i, 4));
        end
    end
    fprintf('      Mean       %f        %f         %f          %f\n\n', heldout_special3_accu(11, 1), heldout_special3_accu(11, 2), heldout_special3_accu(11, 3), heldout_special3_accu(11, 4));
    fprintf('Chosen coefficient for dataset3: %f\n\n', special_lambda_chosen3);
    
    %% print all accuracies for dataset4
    fprintf('Choosing optimal coefficient for dataset4:\n');
    fprintf('Cross Validation Accuracies For Different grid sizes:\n\n');
    fprintf('    accuracy         1              0.1              0.01             0.001\n');
    for i = 1 : 10
        if i ~= 10
            fprintf('      No.%d       %f        %f         %f          %f\n', i, heldout_special4_accu(i, 1), heldout_special4_accu(i, 2), heldout_special4_accu(i, 3), heldout_special4_accu(i, 4));
        else 
            fprintf('      No.%d      %f        %f         %f          %f\n', i, heldout_special4_accu(i, 1), heldout_special4_accu(i, 2), heldout_special4_accu(i, 3), heldout_special4_accu(i, 4));
        end
    end
    fprintf('      Mean       %f        %f         %f          %f\n\n', heldout_special4_accu(11, 1), heldout_special4_accu(11, 2), heldout_special4_accu(11, 3), heldout_special4_accu(11, 4));
    fprintf('Chosen coefficient for dataset4: %f\n\n', special_lambda_chosen4);
    
    fprintf('Pairwise regularization summarize:\n\n');
    fprintf('    Statistics     train_accu     heldout_accu     test_accu      chosen_lambda\n');
    fprintf('    dataset 1       %f        %f        %f           %f\n', special_train_accu1, max(heldout_special1_accu(11, :)), special_test_accu1, special_lambda_chosen1);
    fprintf('    dataset 2       %f        %f        %f           %f\n', special_train_accu2, max(heldout_special2_accu(11, :)), special_test_accu2, special_lambda_chosen2);
    fprintf('    dataset 3       %f        %f        %f           %f\n', special_train_accu3, max(heldout_special3_accu(11, :)), special_test_accu3, special_lambda_chosen3);
    fprintf('    dataset 4       %f        %f        %f           %f\n\n', special_train_accu4, max(heldout_special4_accu(11, :)), special_test_accu4, special_lambda_chosen4);
    fprintf('====================================================================================\n\n');
    

    
    %% plotting parameters for dataset2 & 3 and compare parameters with & without pairwise
    
    [theta2_special, J2_special, exit_flag] = fminunc(@(t)(costFunctionPairWise(t, x2Trl2_bias, y2, special_lambda_chosen2, S)), initTheta2, options);
    [theta3_special, J3_special, exit_flag] = fminunc(@(t)(costFunctionPairWise(t, x3Trl2_bias, y3, special_lambda_chosen3, S)), initTheta3, options);
    
    theta2_batch = transferToMatrix(theta2_batch);
    theta3_batch = transferToMatrix(theta3_batch);
    theta2_special = transferToMatrix(theta2_special);
    theta3_special = transferToMatrix(theta3_special);
    
    fprintf('Plotting 3D-bar with pairwise regularization for dataset2 & dataset3...\n');
    
    fig_theta2_batch = figure(10);
    figure(fig_theta2_batch);
    bar3(theta2_batch);
    title('batch dataset2','FontSize',15,'FontWeight','Demi');
    fig_theta2_special = figure(11);
    figure(fig_theta2_special);
    bar3(theta2_special);
    title('pairwise dataset2','FontSize',15,'FontWeight','Demi');
    fig_theta3_batch = figure(12);
    figure(fig_theta3_batch);
    bar3(theta3_batch);
    title('batch dataset3','FontSize',15,'FontWeight','Demi');
    fig_theta3_special = figure(13);
    figure(fig_theta3_special);
    bar3(theta3_special);
    title('pairwise dataset3','FontSize',15,'FontWeight','Demi');
    
    
    
    %% Using normalized batch gradient descent to train datasets
%     
%     % Optimize
%     [theta1, J1, ac1] = batchGDL2(initTheta1, x1Trl2_bias, y1, 1, 500, lambdas(1));
%     accu = calculateAccuracy(theta1, x1Tel2_bias, y1Te);
%     [theta2, J1, ac1] = batchGDL2(initTheta2, x2Trl2_bias, y2, 1, 500, lambdas(2));
%     accu2 = calculateAccuracy(theta2, x2Tel2_bias, y2Te);
%     [theta3, J1, ac1] = batchGDL2(initTheta3, x3Trl2_bias, y3, 1, 500, lambdas(3));
%     accu3 = calculateAccuracy(theta3, x3Tel2_bias, y3Te);
%     [theta4, J1, ac1] = batchGDL2(initTheta4, x4Trl2_bias, y4, 1, 500, lambdas(4));
%     accu4 = calculateAccuracy(theta4, x4Tel2_bias, y4Te);
    
    %% Visualization of parameters for dataset2 & dataset3
    
    
    
end