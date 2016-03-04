function [X] = normalize(dataX)
% NORMALIZE function normalize the input data
    X = zeros(size(dataX));

    for i = 1:length(dataX(1, :))
        
        m = mean(dataX(:, i));
        s = sqrt(var(dataX(:, i)));
        
        if s ~= 0
            X(:, i) = (dataX(:, i) - mean(dataX(:, i))) / sqrt(var(dataX(:, i)));
        else
            X(:, i) = (dataX(:, i) - mean(dataX(:, i)));
        end
        
    end

end