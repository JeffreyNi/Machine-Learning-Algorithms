function accus = calculateCrossAccu(x, y)

    [m n] = size(x);

    sz = m / 10;
    accus = zeros(11, 1);

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
%         initTheta = zeros(size(x,2)+1, 1);
%         [theta J accuTr] = batchGD(initTheta, [ones(size(train, 1), 1) train], trainy, 1, 500);
        accus(i, 1) = calculateAccuracy(theta, [ones(sz, 1) cross], crossy);
    end
    
    accus(11, 1) = mean(accus(1:10, :));


end