function plotLRlineWithLambda(lambda, X, y, fig)
% [X, y]: contains the data of the samples
% lambda: weight decay parameter

    len = length(lambda);
    
    for i = 1:len
        lam = lambda(i);
        w = ridgeNormalEqn(X, y, lam);
   
        bl = [min(X(:,2))-1, min(y)-1];
        ur = [max(X(:,2))+1, max(y)+1];
        figure(fig)
        subplot(2,2,i), plot(X(1:end-2,2), y(1:end-2),'o','MarkerSize',8,'LineWidth',2);
        hold on, 
        plot(X(end-1,2), y(end-1),'go','MarkerSize',8,'LineWidth',2);
        plot(X(end,2), y(end),'ro','MarkerSize',8,'LineWidth',2);
        plot([bl(1), ur(1)], [w'*[1;bl(1)], w'*[1;ur(1)]],'k-','lineWidth',3);
        title(['(lambda = ' num2str(lam), ')'],'FontSize',12,'FontWeight','Demi')
        xlabel('x','FontSize',15,'FontWeight','Demi')
        ylabel('y','FontSize',15,'FontWeight','Demi')
    end

end