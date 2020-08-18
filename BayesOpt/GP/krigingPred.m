function [mu,mse] = krigingPred(x,xT,yT,beta,sigma,theta,K,deg)
    FT = polymat(xT,deg); 
    k = kernel(theta,xT,x);
    f = polymat(x,deg);
    mu = f*beta + k'*(K\(yT-FT*beta));
    mse = diag((sigma*(1 - k'*(K\k)))^2);
    %mse = diag(sigma*(1-k'*(K\k))).^2;
end

