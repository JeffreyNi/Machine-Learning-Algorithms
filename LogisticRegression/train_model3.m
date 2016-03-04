function [train_accu test_accu] = train_model3(x1_train, x2_train, x3_train, x1_test, x2_test, x3_test)

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
    
    B = mnrfit(x_train, y_train);

    y_model3 = estimateMNL(x_train, B);
    
    train_accu = length(find(y_model3 == y_train)) / ttl_size;
    
    y_test_model3 = estimateMNL(x_test, B);
    
    test_accu = length(find(y_test_model3 == y_test)) / ttl_size_test;

    
end

