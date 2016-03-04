function [X Y] = separateLabel(data)

X = data(:, 1:end-1);
Y = data(:, end);

end