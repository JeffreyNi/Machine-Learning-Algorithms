function plotSamples(figureLR, subPID, dataX, dataY, h, t, d)
% h, t, d: 
%   leverage, studentized_residual, cook's distance of samples
    [~, idxH] = max(h);
    [~, idxT] = max(t);
    [~, idxD] = max(d);
    
    dataY = dataY(:);
    if size(dataX,1)~=length(dataY)
        dataX=dataX';
    end
    if size(dataX,1)~=length(dataY)
        error('data size inconsistent');
    end
    
    figure(figureLR)
    subplot(2,2,subPID), plot(dataX(1:end-2,2), dataY(1:end-2),'o','MarkerSize',8,'LineWidth',2);
    hold on, 
    plot(dataX(end-1,2), dataY(end-1),'go','MarkerSize',8,'LineWidth',2);
    plot(dataX(end,2), dataY(end),'ro','MarkerSize',8,'LineWidth',2);
    plot(dataX(idxH(1),2), dataY(idxH(1)), 'cd','MarkerSize',12,'LineWidth',3);
    plot(dataX(idxT(1),2), dataY(idxT(1)), 'ms','MarkerSize',12,'LineWidth',3);
    plot(dataX(idxD(1),2), dataY(idxD(1)), 'k>','MarkerSize',12,'LineWidth',3);
    hleg = legend('majority','leverage','outlier','max-H','max-T','max-Cook');
    set(hleg,'Location','NorthWest','FontSize',12,'FontWeight','Demi');
    title(['(', num2str(subPID), ')'],'FontSize',12,'FontWeight','Demi')
    xlabel('x','FontSize',15,'FontWeight','Demi')
    ylabel('y','FontSize',15,'FontWeight','Demi')
end