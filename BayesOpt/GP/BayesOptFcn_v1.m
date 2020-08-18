function [xT,yT] = BayesOptFcn_v1(fun,kappa,x0,maxit,sigmaY,paropt,verbose)

    % starting point for bayes opt
    if isempty(x0)
        x0 = rand(1,2);
    end
    % max number of bayes opt iterations
    if isempty(maxit)
        maxit = 20;
    end
    % inherent data noise
    if isempty(sigmaY)
        sigmaY = 0;
    end
    % length scale optimization parameters
    if isempty(paropt)
        paropt.nruns = 200; paropt.niter = 1e3; 
        paropt.alpha = 0.01; paropt.beta = 1; 
        paropt.normalize = 0;
        paropt.lb = []; paropt.ub = [];
    end

    % begin bayes opt iterations
    xT = x0;
    yT = feval(fun,x0);
    i = 0;
    while i <= maxit
        
        % check for pos def stopping criterion
        if i > 0
            [K,~] = kernfcn(xT,xT,theta);
            if sigmaY == 0
                [~,flag] = chol(K+1e-15*eye(length(K)),'lower');
            else
                [~,flag] = chol(K+sigmaY^2*eye(length(K)),'lower');
            end
            if flag == 1
                return
            end
        end
        
        if verbose == 1
            fprintf('Iteration-%d; FunEval = %4.2f\n',i,yT(end))
        end
     
        % length scale optimization
        mle = @(x) mlefun(xT,yT,x,sigmaY);
        jmle = @(x) jacmlefun(xT,yT,x,sigmaY);
        x0 = (3-0.05)*rand(1,size(xT,2)) + 0.05;
        [theta,~] = cg_optim_wrap(mle,jmle,x0,paropt.lb,paropt.ub,paropt.alpha,paropt.beta,paropt.nruns,paropt.niter,paropt.normalize);
        
        % optimize acquisition function to select next point
        A = @(x) -acquisition_fcn(x,xT,yT,kappa,sigmaY,theta);
        x0 = rand(1,size(xT,2));
        [xsolve,~] = fminsearch(A,x0);
        xT = [xT;xsolve];
        yT = [yT;feval(fun,xsolve)];
        i = i+1;
    end

end

function [A,flag] = acquisition_fcn(x,xTrain,yTrain,kappa,sigmaY,theta)
   [Mu,Cov,flag] = gprfcn(x,xTrain,yTrain,sigmaY,theta);
    A = Mu - kappa*sqrt(diag(Cov));
end

function [Mu,Cov,flag] = gprfcn(xTest,xTrain,yTrain,sigmaY,theta)
    [K,~] = kernfcn(xTrain,xTrain,theta);
    [Ks,~] = kernfcn(xTest,xTrain,theta);
    [Kss,~] = kernfcn(xTest,xTest,theta);
    
    if sigmaY == 0
        [L,flag] = chol(K+eps*length(K)*eye(length(K)),'lower');
    else
        [L,flag] = chol(K+sigmaY^2*eye(length(K)),'lower');
    end
    
    alpha = L'\(L\yTrain);
    theta1 = 1/length(K)*yTrain'*alpha;
    Mu = Ks*alpha;
    v = L\(Ks');
    Cov = theta1*(Kss-v'*v); Cov(Cov<1e-60) = 0;
end

function [logL] = mlefun(xTrain,yTrain,theta,sigmaY)
    [K,~] = kernfcn(xTrain,xTrain,theta);
    if sigmaY == 0
        L = chol(K+1e-15*length(K)*eye(length(K)),'lower');
    else
        L = chol(K+sigmaY^2*eye(length(K)),'lower');
    end
    alpha = L'\(L\yTrain);
    theta1 = 1/length(K)*yTrain'*alpha;
    logL = -length(K)/2*(1 + log(2*pi) + log(theta1)) - sum(log(diag(L)));
    logL = -logL;
end

function [JlogL] = jacmlefun(xTrain,yTrain,theta,sigmaY)

    [K,Daa] = kernfcn(xTrain,xTrain,theta);
    if sigmaY == 0
        L = chol(K+1e-15*length(K)*eye(length(K)),'lower');
    else
        L = chol(K+sigmaY^2*eye(length(K)),'lower');
    end

    alpha = L'\(L\yTrain); 
    theta1 = 1/length(K)*yTrain'*alpha;
    JlogL = zeros(1,length(theta));
    
    for i = 1:length(theta)
        dKdthi = 1/theta(i)^3*Daa{i}.*(L*L');
        Temp = L'\(L\dKdthi);
        JlogL(i) = 1/(2*theta1)*(yTrain'*Temp*alpha) - 1/2*trace(Temp);
    end
    JlogL = -JlogL;
end

function [K,Daa] = kernfcn(X,Y,theta)
    D = zeros(size(X,1),size(Y,1));
    Daa = cell(1,length(theta));
    for i = 1:length(theta)
        Da = distance_matrix(X(:,i)',Y(:,i)');
        D = D + Da./(theta(i)^2);
        Daa{i} = Da;
    end
    K = exp(-1/2*D);
end

function [D] = distance_matrix(X, Y, varargin)
    % X,Y are nfeatures x nsamples
    % column vectors
    if isempty(varargin)
        p = 2;
    else
        p = varargin{1};
    end
    if size(X) == size(Y)
        if strcmp(p,'Inf')
            % Requires X,Y to be N x K, N = # points, K = # dimensions!
            X = permute(X,[1,2,3]);
            Y = permute(Y,[3,2,1]);
            D = squeeze(max(abs(bsxfun(@minus, X, Y)),[],2));
        else
            D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
        end
    else
        D = zeros(size(X,2),size(Y,2));
        for i = 1:size(X,2)
            for j = 1:size(Y,2)
                D(i,j) = sum( (X(:,i) - Y(:,j)).^2 );
            end
        end
    end
end

function [x_hist,f_hist,jf_hist] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize)
    x_hist = zeros(niter,length(x0)); x_hist(1,:) = x0;
    f_hist = zeros(niter,1); f_hist(1) = mle(x0);
    jf_hist = zeros(niter,length(x0)); jf_hist(1,:) = jmle(x0);
    dtemp = jf_hist(1,:);
    k = 2;
    flag = 0;
    while k <= niter 
        jf_hist(k,:) = jmle(x_hist(k-1,:));
        if normalize == 1 % normalize by total norm
            d = (1-beta)*dtemp + beta*jf_hist(k,:)/norm(jf_hist(k,:));
        elseif normalize == 2 % normalize by component norm
            d = (1-beta)*dtemp + beta*sign(jf_hist(k,:))*sqrt(length(x0));
        else % no normalization
            d = (1-beta)*dtemp + beta*jf_hist(k,:);
        end
        x_hist(k,:) = x_hist(k-1,:) - alpha*d;
        f_hist(k) = mle(x_hist(k,:));
        dtemp = d;
        if norm(x_hist(k,:) - x_hist(k-1,:)) <= eps
            flag = 1;
        end
        k = k + 1;
     end
end

function [xopt,fopt] = cg_optim_wrap(mle,jmle,x0,lb,ub,alpha,beta,nruns,niter,normalize)
    mm = 10;
    
    [x1_hist,f1_hist,~] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize);
    f0 = mean(f1_hist(end-mm:end));
    xopt = x1_hist(end,:);
    
    for i = 2:nruns
        if isempty(lb) && isempty(ub)
            x0 = rand(1,length(x0));
        else
            x0 = (ub-lb).*rand(1,length(x0)) + lb;
        end
        [x1_hist,f1_hist,~] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize);
        fopt = mean(f1_hist(end-50:end));
        if fopt < f0
            xopt = x1_hist(end,:);
            f0 = fopt;
        end
    end
end