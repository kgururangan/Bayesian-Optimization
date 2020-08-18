function [beta,sigma] = calcBetaSigma(xT,yT,K,deg)
    FT = polymat(xT,deg); 
    beta  = (FT'*(K\FT))\(FT'*(K\yT));
    sigma = 1/length(xT)*(yT-FT*beta)'*(K\(yT-FT*beta));
    %sigma = 1/length(yT)*(yT-FT*beta)'*inv(K)*(yT-FT*beta);
end

