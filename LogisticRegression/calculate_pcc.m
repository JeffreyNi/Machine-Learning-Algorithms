function pcc = calculate_pcc(X, Y)

[m n] = size(X);
pcc = zeros(n, 1);

for i = 1 : n
   
    partX = X(:, i);
    C = cov(partX, Y);
    pcc(i, 1) = C(1, 2) / sqrt(var(partX) * var(Y));
    
end


end