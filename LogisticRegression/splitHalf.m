function [first second] = splitHalf(data)
% split the matrix into two equal half
numRows = size(data, 1);
numFirst = round(numRows/2);

first = data(1:numFirst, :);
second = data(numFirst + 1:end, :);

end