function matrix = transferToMatrix(theta)

sz = 8;
matrix = zeros(sz, sz);

for i = 0 : (sz-1)
   
    matrix(i + 1, :) = theta((i*sz+1): (i*sz+sz), :)';
    
end

end