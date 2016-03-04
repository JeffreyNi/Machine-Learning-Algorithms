function newX = singleDiscretize(x, numBin)

    m = length(x);
    newX = zeros(m, numBin);
    
    xMin = min(x);
    xMax = max(x);
    stepSZ = (xMax - xMin) / numBin;
    theRange = xMin : stepSZ : xMax;
    
    for i = 1 : m
        bin_num = getRange(theRange, x(i));
        newX(i, bin_num) = 1;
    end
    

end