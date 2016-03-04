function plotLRline(figureLR, subPID, w1, w2, dataX, dataY)
% [dataX, dataY]: coordinates of data samples
% w: the linear regresison parameters

    w1 = transpose(w1(:));
    w2 = transpose(w2(:));
    dataY = dataY(:);
    if size(dataX,1)~=length(dataY)
        dataX=dataX';
    end
    if size(dataX,1)~=length(dataY)
        error('data size inconsistent');
    end
    
    bl = [min(dataX(:,2))-1, min(dataY)-1];
    ur = [max(dataX(:,2))+1, max(dataY)+1];
    figure(figureLR)
    subplot(2,2,subPID), plot(dataX(1:end-2,2), dataY(1:end-2),'o','MarkerSize',8,'LineWidth',2);
    hold on, 
    plot(dataX(end-1,2), dataY(end-1),'go','MarkerSize',8,'LineWidth',2);
    plot(dataX(end,2), dataY(end),'ro','MarkerSize',8,'LineWidth',2);
    plot([bl(1), ur(1)], [w1*[1;bl(1)], w1*[1;ur(1)]],'k-','lineWidth',3);
    plot([bl(1), ur(1)], [w2*[1;bl(1)], w2*[1;ur(1)]],'m-','lineWidth',3);
    hleg = legend('majority','leverage','outlier','LR with outlier', 'LR without outlier');
    set(hleg,'Location','NorthWest','FontSize',12,'FontWeight','Demi');
    title(['(' num2str(subPID), ')'],'FontSize',12,'FontWeight','Demi')
    xlabel('x','FontSize',15,'FontWeight','Demi')
    ylabel('y','FontSize',15,'FontWeight','Demi')
end