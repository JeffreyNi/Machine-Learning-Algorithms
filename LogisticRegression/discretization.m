function [x] = discretization(o, sz, range)

    step = (range(2) - range(1)) / sz;
    
    m = size(o, 1);
    
    ranges = range(1) : step : range(2);
    
    num = sz^2;
    x = zeros(m, num);  
    
    for i = 1 : m
        
        col = getRange(ranges, o(i, 1));
        row = getRange(ranges, o(i, 2));
        NO = (row - 1) * sz + col;
        x(i, NO) = 1;
        
    end

end