function [newdata] = randomShuffle(data) 
% random shuffles the rows of a matrix
numRow = size(data, 1);
randomRows = randperm(numRow);

newdata = data(randomRows, :);

end