function [new_accu, train_accu, dist] = Knn_classify(train_data, train_label, new_data, new_label, K)
% K-nearest neighbor classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%  K: number of nearest neighbors
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data (using leave-one-out
%  strategy)
%
% CSCI 567: Machine Learning, Spring 2016, HomeworK 1

% % standardize data
% train_data = normalize(train_data);
% new_data = normalize(new_data);

train_data = normalize(train_data);
new_data = normalize(new_data);

% initialize accuracy
new_accu = 0;
train_accu = 0;

% compute train accuracy
train_dist = pdist2(train_data, train_data); % comput distance matrix
m = length(train_label);
for i = 1:m

    [sortedRow idx] = sort(train_dist(i, :));
    neighbors = idx(2:K+1); % skip the first one, because it is point itself
    label = mode(train_label(neighbors));
    
    if label == train_label(i)
        train_accu = train_accu + 1;
    end
end
train_accu = train_accu / m;


% compute new accuracy
new_dist = pdist2(new_data, train_data); % compute distance matrix for new_data
new_m = length(new_label);
for i = 1:new_m

    [sortedRow idx] = sort(new_dist(i, :));
    neighbors = idx(1:K);;
    label = mode(train_label(neighbors));

    if label == new_label(i)
        new_accu = new_accu + 1;
    end

end
new_accu = new_accu / new_m;
dist = new_dist;

