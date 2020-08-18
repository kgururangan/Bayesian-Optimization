function [K] = kernel(theta,u,v)
%D = distanceMatrix(u,v);
D = sqrt(pdist_kg(u',v'));
K = exp(-D.^2/(2*theta(1)^2));
end

