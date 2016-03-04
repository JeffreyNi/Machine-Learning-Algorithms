function accu = calculateCrossAccu(x, y)

    [m n] = size(x);
%     rg = randperm(m);
%     x = x(rg, :);
%     y = y(rg);
    sz = m / 10;
%     accus = zeros(10, 1);
    accus = [];

    for i = 1 : 10
        
        if i ~= 10
            cross =  x(((i-1)*sz+1) : (i * sz), :);
            crossy = y(((i-1)*sz+1) : (i * sz), :);
            train = x([1:((i-1)*sz), (i*sz+1):m], :);
            trainy = y([1:((i-1)*sz), (i*sz+1):m], :);
        else
            cross = x(((i-1)*sz+1) : m, :);
            crossy = y(((i-1)*sz+1) : m, :);
            train = x(1:((i-1)*sz) ,:);
            trainy = y(1:((i-1)*sz) ,:);
        end
        
        theta = glmfit(train, trainy, 'binomial', 'link', 'logit');
        acc = calculateAccuracy(theta, [ones(sz, 1) cross], crossy);
        accus = [accus acc];
    end
    
    accu = mean(accus);
    
%     range = 1 : sz;
%     trainX = x((sz+1):end, :);
%     trainY = y((sz+1):end, :);
%     testX = x(range, :);
%     testY = y(range, :);
%     [theta dev stats] = glmfit(trainX, trainY, 'binomial', 'link', 'logit');
%     accus(1) = calculateAccuracy(theta, [ones(sz, 1) testX], testY);
%     
%     for i = 1 : 8
%         
%         prevRange = 1 : (i*sz);
%         afterRange = ((i+1)*sz + 1) : m;
%         trainX = [x(prevRange, :); x(afterRange, :)];
%         trainY = [y(prevRange); y(afterRange)]; 
%         
%         range = (i * sz + 1) : ((i+1)*sz);
%         testX = x(range, :);
%         testY = y(range, :);
%         [theta dev stats] = glmfit(trainX, trainY, 'binomial', 'link', 'logit');
%         accus(i + 1) = calculateAccuracy(theta, [ones(sz, 1) testX], testY);
%         
%         
% %         prevRange
% %         afterRange
% %         range
%     end
% 
%     range = (sz * 9 + 1) : m;
%     trainX = x(1:sz*9, :);
%     trainY = y(1:sz*9, :);
%     testX = x(range, :);
%     testY = y(range, :);
%     [theta dev stats] = glmfit(trainX, trainY, 'binomial', 'link', 'logit');
%     accus(10) = calculateAccuracy(theta, [ones(sz, 1) testX], testY);
%     
%     accu = mean(accus);

end