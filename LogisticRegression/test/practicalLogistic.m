function practicalLogistic()

    data = load('toySpiral.mat');
    data1 = data.data1;
    data2 = data.data2;
    data3 = data.data3;
    data4 = data.data4;
    
    dis_size = [2 4 8 16];
    range = [-2 2];
    
    % dataset 1
    y1 = data1.yTr;
    y1 = y1 - 1;
%     rg1 = find(y1 == 2);
%     y1(rg1) = 0;
    y1Te = data1.yTe;
%     rg1Te = find(y1Te == 2);
%     y1Te(rg1Te) = 0;
    
    cross_accu1 = zeros(1, 4);
    
    for i = 1 : 4
        
        sz = dis_size(i);
        x1 = discretization(data1.xTr, sz, range);
        cross_accu1(i) = calculateCrossAccu(x1, y1);
        
    end
    
    % dataset 2
    y2 = data2.yTr;
    rg2 = find(y2 == 2);
    y2(rg2) = 0;
%     y2Tr = data2.yTr;
%     rg2Tr = find(y2Tr == 2);
%     y2Tr(rg2Tr) = 0;
    y2Te = data2.yTe;
    rg2Te = find(y2Te == 2);
    y2Te(rg2Te) = 0;
    
    cross_accu2 = zeros(1, 4);
    
    for i = 1 : 4
        
        sz = dis_size(i);
        x2 = discretization(data2.xTr, sz, range);
        cross_accu2(i) = calculateCrossAccu(x2, y2);
        
    end
%     
%     % dataset 3
%     y3 = data3.yTr;
%     rg3 = find(y3 == 2);
%     y3(rg3) = 0;
% %     y3Tr = data3.yTr;
% %     rg3Tr = find(y3Tr == 2);
% %     y3Tr(rg3Tr) = 0;
%     y3Te = data3.yTe;
%     rg3Te = find(y3Te == 2);
%     y3Te(rg3Te) = 0;
%     
%     cross_accu3 = zeros(1, 4);
%     
%     for i = 1 : 4
%         
%         sz = dis_size(i);
%         x3 = discretization(data3.xTr, sz, range);
%         cross_accu3(i) = calculateCrossAccu(x3, y3);
%         
%     end
%     
%      % dataset 4
%     y4 = data4.yTr;
%     rg4 = find(y4 == 2);
%     y4(rg4) = 0;
% %     y4Tr = data4.yTr;
% %     rg4Tr = find(y4Tr == 2);
% %     y4Tr(rg4Tr) = 0;
%     y4Te = data4.yTe;
%     rg4Te = find(y4Te == 2);
%     y4Te(rg4Te) = 0;
%     
%     cross_accu4 = zeros(1, 4);
%     
%     for i = 1 : 4
%         
%         sz = dis_size(i);
%         x4 = discretization(data4.xTr, sz, range);
%         cross_accu4(i) = calculateCrossAccu(x4, y4);
%         
%     end



end