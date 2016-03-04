function accu = calculateAccuracy(theta, X, y)

label = round(sigmoid(X * theta));
accu = length(find(label == y)) / length(y);

end