function [D] = distanceMatrix(u,v)
[a1,a2] = size(u); [a3,a4] = size(v);
D = zeros(a1,a3);

    for i = 1:a1
        for j = 1:a3
            D(i,j) = norm(u(i,:) - v(j,:));
        end
    end

end

