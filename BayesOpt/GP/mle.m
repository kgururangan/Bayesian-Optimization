function [L] = mle(theta,xT,yT,deg,epsilon)
    %epsilon = 10^-9;
    N  = length(yT);
    FT = polymat(xT,deg); 
    K = kernel(theta,xT,xT) + epsilon*eye(N); 
    Lchol = chol(K);
    [beta,sigma] = calcBetaSigma(xT,yT,K,deg);
    L = -1/2*log(det(chol(K))) - 1/(2*sigma^2)*(yT-FT*beta)'*(yT-FT*beta) - N/2*log(2*pi*sigma^2);
    %L = -1/2*log(sum(diag(Lchol))) - 1/(2*sigma^2)*(yT-FT*beta)'*(yT-FT*beta) - N/2*log(2*pi*sigma^2);
    L = -L;
end

