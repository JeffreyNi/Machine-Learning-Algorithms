function idx = getRange(ranges, x)
    
    idx = 0;
    n = length(ranges) - 1;
    
    for i = 1 : (n - 1)
       if x >= ranges(i) && x < ranges(i + 1)
           idx = i;
           break;
       end
    end

    if idx == 0
        idx = n;
    end

end