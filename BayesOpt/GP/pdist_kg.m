function D = pdist_kg(X, Y, varargin)

    if isempty(varargin)
        p = 2;
    else
        p = varargin{1};
    end

    if strcmp(p,'Inf')
        % Requires X,Y to be N x K, N = # points, K = # dimensions!
        X = permute(X,[1,2,3]);
        Y = permute(Y,[3,2,1]);
        D = squeeze(max(abs(bsxfun(@minus, X, Y)),[],2));
    else
        D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
    end

end