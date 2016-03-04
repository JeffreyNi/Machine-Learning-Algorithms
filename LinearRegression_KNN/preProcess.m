function [data label] = preProcess(filename)

data = importdata(filename);
origin = data;

num_samples = length(data);
n = length(strsplit(data{1}, ','));

data = zeros(num_samples, 21);
label = zeros(num_samples, 1);

% construct maps
keySet1 = {'vhigh', 'high', 'med', 'low'};
valueSet1 = {[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]};
buying = containers.Map(keySet1, valueSet1);

maint = containers.Map(keySet1, valueSet1);

keySet3 = {'2', '3', '4', '5more'};
doors = containers.Map(keySet3, valueSet1);

keySet4 = {'2', '4', 'more'};
valueSet4 = {[1, 0, 0], [0, 1, 0], [0, 0, 1]};
person = containers.Map(keySet4, valueSet4);

keySet5 = {'small', 'med', 'big'};
boot = containers.Map(keySet5, valueSet4);

keySet6 = {'low', 'med', 'high'};
safety = containers.Map(keySet6, valueSet4);

keySet7 = {'unacc', 'acc', 'good', 'vgood'};
class = containers.Map(keySet7, [1, 2, 3, 4]);

% build data matrix and label vector
for i = 1:num_samples
    
    row = strsplit(origin{i}, ',');
    label(i) = class(char(row(7)));
    
    key1 = char(row(1));
    data(i, 1:4) = buying(key1);
    key2 = char(row(2));
    data(i, 5:8) = maint(key2);
    key3 = char(row(3));
    data(i, 9:12) = doors(key3);
    key4 = char(row(4));
    data(i, 13:15) = person(key4);
    key5 = char(row(5));
    data(i, 16:18) = boot(key5);
    key6 = char(row(6));
    data(i, 19:21) = safety(key6);
    
end
