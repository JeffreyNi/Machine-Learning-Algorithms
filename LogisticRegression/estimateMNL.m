function label = estimateMNL(X, B)

[m n] = size(X);
num_label = size(B, 2);

label = zeros(m, 1);

for i = 1 : m
   
    ps = zeros(num_label + 1, 1);
    
    for j = 1 : num_label
        ps(j) = exp([1 X(i, :)] * B(:, j));
    end
    ps(num_label + 1) = 1;

    [val label(i)] = max(ps);
end

end