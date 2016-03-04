function [train_accu, test_accu] = model1(x1_train, x2_train, x3_train, x1_test, x2_test, x3_test)

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

    S = sqrt(var(x_train));
    S1 = S;
    S2 = S;
    S3 = S;
    
    y_model2 = zeros(ttl_size, 1);
    
    for i = 1 : ttl_size
        
        p1 = calculateGaussian(x_train(i, :), mu1, S1) * pi1;
        p2 = calculateGaussian(x_train(i, :), mu2, S2) * pi2;
        p3 = calculateGaussian(x_train(i, :), mu3, S3) * pi3;
        p = [p1 p2 p3];
        [val y_model2(i)] = max(p);
        
    end
    
    train_accu = length(find(y_model2 == y_train)) / ttl_size;
    
    y_test_model2 = zeros(ttl_size_test, 1);
    
    for i = 1 : ttl_size_test
        
       p1 = calculateGaussian(x_test(i, :), mu1, S1) * pi1;
       p2 = calculateGaussian(x_test(i, :), mu2, S2) * pi2;
       p3 = calculateGaussian(x_test(i, :), mu3, S3) * pi3;
       p = [p1 p2 p3];
       [val y_test_model2(i)] = max(p);
    
    end
    
    test_accu = length(find(y_test_model2 == y_test)) / ttl_size_test;


end