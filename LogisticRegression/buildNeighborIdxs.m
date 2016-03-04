function S = buildNeighborIdxs(grid)

S = [];

for i = 1 : (grid - 1)
    for j = 1 : (grid - 1)
        num = (i - 1) * grid + j;
        pair1 = [num num+1];
        pair2 = [num num+grid];
        S = [S;pair1];
        S = [S;pair2];
    end
        num = num + 1;
        pair = [num num+grid];
        S = [S;pair];
end

for j = 1 : (grid - 1)
    num = (grid - 1)*grid + j;
    pair = [num num+1];
    S = [S; pair];
end


end