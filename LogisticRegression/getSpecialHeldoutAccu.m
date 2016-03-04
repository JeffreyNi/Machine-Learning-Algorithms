function accus = getSpecialHeldoutAccu(x, y, lambda, S)

    [m n] = size(x);

    sz = m / 10;
    accus = zeros(11, 1);
    options = optimset('display', 'off', 'MaxIter', 400);

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
        
        initTheta = zeros(size(train, 2), 1);
        
        [theta, J, exit_flag] = fminunc(@(t)(costFunctionPairWise(t, train, trainy, lambda, S)), initTheta, options);
%         [theta, J, accu] = batchGDL2(initTheta, train, trainy, 1, 100, lambda);
        accus(i, 1) = calculateAccuracy(theta, cross, crossy);
    end
    
    accus(11, 1) = mean(accus(1:10, :));
    


end