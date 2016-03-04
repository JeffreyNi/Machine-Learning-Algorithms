function lrTemplate()
    load dataHW1LR.mat
    
    %%
    % your codes here
    
    %%
    %     demo of TA's plot functions
    %     figureLR = figure(1);
    %     set(figureLR,'Position',[100, 100, 1000, 800]);
    %     plotLRline(figureLR, 1, w1A, w1B, data1.x, data1.y);
    %     plotLRline(figureLR, 2, w2A, w2B, data2.x, data2.y);
    %     plotLRline(figureLR, 3, w3A, w3B, data3.x, data3.y);
    %     plotLRline(figureLR, 4, w4A, w4B, data4.x, data4.y);
    %     
    %     figureSample = figure(2);
    %     set(figureSample,'Position',[100, 100, 1000, 800]);
    %     plotSamples(figureSample, 1, data1.x, data1.y, h1, t1, d1);
    %     plotSamples(figureSample, 2, data2.x, data2.y, h2, t2, d2);
    %     plotSamples(figureSample, 3, data3.x, data3.y, h3, t3, d3);
    %     plotSamples(figureSample, 4, data4.x, data4.y, h4, t4, d4);
end

function r = heldoutResidual(w, dataValX, dataValY)
% evaluate/calculate the residual on heldout data
% your code here

end

function w = lrFit(dataX,dataY)
% fit the linear regression parameters    
% your code here
w = pinv(dataX' * dataX) * dataX' * dataY;
end

function h = leverage(dataX)
% calculate leverage of every sample    
% your code here
end

function t = studentized(dataX,dataY,w,h)
% calculate studentized residual for every sample
% your code here
end

function d = cookDist(h,t,k)
% calculate cook's distance for every sample
% your code here
end

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
    hleg = legend('majority','leverage','outlier','LR');
    set(hleg,'Location','NorthWest','FontSize',12,'FontWeight','Demi');
    title(['(' num2str(subPID), ')'],'FontSize',12,'FontWeight','Demi')
    xlabel('x','FontSize',15,'FontWeight','Demi')
    ylabel('y','FontSize',15,'FontWeight','Demi')
end

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
