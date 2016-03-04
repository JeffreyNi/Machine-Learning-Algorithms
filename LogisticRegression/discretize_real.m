function mi = discretize(X, Y)

mi = 0;
m = length(Y);
size = round(m / 10);
[sortedX idx] = sort(X);
sortedY = Y(idx);

ttl0 = length(find(Y == 0));
py0 = ttl0 / m;
py1 = 1 - py0;

for i = 0 : 9
    
    if i == 9
        range = (i * size) : m;
        size = length(range);
    else    
        range = (i * size + 1) : (i * size + size);
    end
    
    partX = sortedX(range);
    partY = sortedY(range);
    
    px = size / m;
    num0 = length(find(partY == 0));
    num1 = size - num0;
    pxy0 = num0 / m;
    pxy1 = num1 / m;
    
    %mi = mi + pxy0 * log2(pxy0 / (px * py0)) + pxy1 * log2(pxy1 / (px * py1));
    if pxy0 ~= 0
        mi = mi + pxy0 * log2(pxy0 / (px * py0));
    end
       
    if pxy1 ~= 0
        mi = mi + pxy1 * log2(pxy1 / (px * py1));
    end
    
end


end