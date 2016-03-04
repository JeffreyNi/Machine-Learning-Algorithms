function mi = discretize(X, Y)

mi = 0;
m = length(Y);

MIN = min(X);
MAX = max(X);

ttl0 = length(find(Y == 0));
py0 = ttl0 / m;
py1 = 1 - py0;

for i = MIN : MAX
    
   range = find(X == i);
   size = length(range);
   
   if size ~= 0
       partY = Y(range);
       
       px = size / m;
       num0 = length(find(partY == 0));
       num1 = size - num0;
       pxy0 = num0 / m;
       pxy1 = num1 / m;
       
       if pxy0 ~= 0
           mi = mi + pxy0 * log2(pxy0 / (px * py0));
       end
       
       if pxy1 ~= 0
           mi = mi + pxy1 * log2(pxy1 / (px * py1));
       end
       
   end
    
end


end