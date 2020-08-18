function F = polymat(x,deg)
[N,C] = size(x);
if C == 1
    if deg == 0
        F = ones(N,1);
    elseif deg == 1
        F = [ones(N,1), x(:,1)];
    else
        F = [ones(N,1), x(:,1), x(:,1).^2];
    end
elseif C == 2  
    if deg == 0
        F = ones(N,1);
    elseif deg == 1
        F = [ones(N,1), x(:,1), x(:,2)];
    elseif deg == 2
        F = [ones(N,1), x(:,1), x(:,2), x(:,1).^2, x(:,2).^2];
    else
        F = [ones(N,1), x(:,1), x(:,2), x(:,1).^2, x(:,2).^2, x(:,1).*x(:,2)];
    end
else
    if deg == 0
        F = ones(N,1);
    elseif deg == 1
        F = [ones(N,1), x(:,1), x(:,2), x(:,3)];
    elseif deg == 2
        F = [ones(N,1), x(:,1), x(:,2), x(:,3), x(:,1).^2, x(:,2).^2, x(:,3).^2];
    else
        F = [ones(N,1), x(:,1), x(:,2), x(:,3), x(:,1).^2, x(:,2).^2, x(:,3).^2, x(:,1).*x(:,2), x(:,2).*x(:,3), x(:,1).*x(:,3)];
    end
end

end
    