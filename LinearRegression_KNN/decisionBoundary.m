function [X y]= decisionBoundary(data, label, k)

    X = zeros(10000, 2);
    y = zeros(10000, 1);
    
    for i = 0:99
       for j = 0:99
          X(i * 100 + j + 1, 1) = 0.01 * j + 0.005;
          X(i * 100 + j + 1, 2) = 0.01 * i + 0.005;
       end
    end
    
    dist = pdist2(X, data);
    for i = 1:10000
       
        [sortedRow idx] = sort(dist(i, :));
        neighbors = idx(1:k);
        y(i) = mode(label(neighbors));
        
    end
    
    newfigure = figure(figure);
    hold on;
    pos_idx = find(y == 1);
    neg_idx = find(y == -1);
    
%     for i = 1:length(pos_idx)
%    
%         plot(X(pos_idx(i), 1), X(pos_idx(i), 2), 'b*','MarkerSize',8);
%         
%     end
%     
%     for i = 1:length(neg_idx)
%        
%         plot(X(neg_idx(i), 1), X(neg_idx(i), 2), 'y*','MarkerSize',8);
%         
%     end
    plot(X(pos_idx, 1), X(pos_idx, 2), 'b*');
    plot(X(neg_idx, 1), X(neg_idx, 2), 'y*');
    title(['(K = ' num2str(k), ')'],'FontSize',12,'FontWeight','Demi')
    xlabel('x','FontSize',15,'FontWeight','Demi')
    ylabel('y','FontSize',15,'FontWeight','Demi')

end