function p = calculateGaussian(x, mu, S)

sigma = diag(S.^2);

p = (1 / (sqrt(2 * pi) * sum(S))) * exp(- (x - mu) * pinv(sigma) * (x - mu)' / 2);

end