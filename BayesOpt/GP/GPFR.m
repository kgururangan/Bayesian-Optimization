%% Gaussian Process Functional Regression (GPFR) workspace
clear all
clc
close all

% simple 1d heat conduction
dx = 0.01;
x = -1:dx:1;
A = -2*diag(ones(length(x),1)) + diag(ones(length(x)-1,1),1) + diag(ones(length(x)-1,1),-1); A = -A;
f = sin(pi*x') + 4*sin(4*pi*x');
u = (A\f)*(2*dx^2);

sigmaY = 0;
nT = 10; 
nTest = length(x)-nT;
idx = randperm(length(x));
xTrain = x(idx(1:nT))'; yTrain = u(idx(1:nT));
xTest = x(idx(nT+1:end))'; yTest = u(idx(nT+1:end));

mle = @(x) mlefun(xTrain,yTrain,x,sigmaY); jmle = @(x) jacmlefun(xTrain,yTrain,x,sigmaY);
lb = 0; ub = 20; alpha = 0.01; beta = 1; nruns = 10; niter = 1e4; normalize = 0;
[theta,~] = cg_optim_wrap(mle,jmle,rand(1),lb,ub,alpha,beta,nruns,niter,normalize);
[Mu,Cov,~,~,~,~,~,~,~] = gprfcn(x',xTrain,yTrain,sigmaY,theta);
%%
fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C );
figure(1)
plot(x,u,'color',[0,0,1],'Linewidth',2); 
set(gca,'FontSize',15,'Linewidth',2,'Box','off'); xlabel('x'); ylabel('u(x)'); grid on;
hold on
plot(x',Mu,'color',[0.8,0,0],'Linewidth',2); h = fill_between_lines(x,(Mu+2*sqrt(diag(Cov)))',(Mu-2*sqrt(diag(Cov)))',[0.3010, 0.7450, 0.9330]);
h.FaceAlpha = 0.3; scatter(xTrain,yTrain,50,'MarkerFaceColor',[0.8,0,0],'MarkerEdgeColor',[0.8,0,0])
ll=legend('u_{True}','GP Mean','95% Confidence','Training Points'); set(ll,'FontSize',15,'Location','NorthWest');
%% functions 
function [A] = acquisition_fcn(x,xTrain,yTrain,kappa,sigmaY,theta)
   [Mu,Cov,~,~,~,~,~,~,~,~] = gprfcn(x,xTrain,yTrain,sigmaY,theta);
   
   % lower confidence bound
   %A = Mu - kappa*sqrt(diag(Cov));
   
   % expected improvement
   % note for minimization of f, we use 
   % Z = (max(yTrain)-Mu(ind)+kappa)/Sig(ind)
   A = zeros(1,length(Mu));
   Sig = sqrt(diag(Cov));
   ind = find(Sig > 0); 
   phi = @(x) pdf('normal',x,0,1);
   Phi = @(x) cdf('normal',x,0,1);
   Z = (Mu(ind) - max(yTrain) - kappa)./Sig(ind);
   A(ind) = Sig(ind).*(Z.*Phi(Z) + phi(Z));
end

function [Mu,Cov,theta,logL,K,Ks,Kss,L,alpha,v] = gprfcn(xTest,xTrain,yTrain,sigmaY,theta)

    [K,~] = kernfcn(xTrain,xTrain,theta);
    [Ks,~] = kernfcn(xTest,xTrain,theta);
    [Kss,~] = kernfcn(xTest,xTest,theta);
    
    if sigmaY == 0
        L = chol(K+eps*length(K)*eye(length(K)),'lower');
    else
        L = chol(K+sigmaY^2*eye(length(K)),'lower');
    end
    alpha = L'\(L\yTrain);
    theta1 = 1/length(K)*yTrain'*alpha;
    %Ks = theta1*Ks; Kss = theta1*Kss;
    %alpha = 1/theta1*alpha;
    
    Mu = Ks*alpha;
    v = L\(Ks');
    Cov = theta1*(Kss-v'*v); Cov(Cov<1e-60) = 0;
    logL = -length(K)/2*(1 + log(2*pi) + log(theta1)) - sum(log(diag(L)));
    logL = -logL;
    %Mu = Ks*alpha;
    %v = 1/sqrt(theta1)*L\(Ks');
    %Cov = Kss - v'*v;  Cov(Cov<1e-60) = 0;
    % are we off by a sqrt(theta1) in the sum(log(diag(L))) term?
    %logL = -1/2*yTrain'*alpha - sum(log(diag(L))) - log(theta1) - length(K)/2*log(2*pi);
    %logL = -logL;

end

function [logL] = mlefun(xTrain,yTrain,theta,sigmaY)
    % use analytical gradient:
    % d(logL)/dx_j = 1/2*Tr( (alpha*alpha' - K^-1)*dK/dx_j )
    % alpha = K^-1*y = inv(L')*inv(L)*y
    % theta(1) = 1/length(K)*yTrain'*(K'\yTrain) where K' = K/theta(1)
    
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
    %mle = @(x) mlefun(xTrain,yTrain,[1,x],sigmaY);
    %jmle = @(x) jacmlefun(xTrain,yTrain,[1,x],sigmaY);
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
    %[f0,i0] = min(f1_hist);
    f0 = mean(f1_hist(end-mm:end));
    xopt = x1_hist(end,:);
    
    for i = 2:nruns
        if isempty(lb) && isempty(ub)
            x0 = rand(1,length(x0));
        else
            x0 = (ub-lb).*rand(1,length(x0)) + lb;
        end
        [x1_hist,f1_hist,~] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize);
        %[fopt,iopt] = min(f1_hist);
        fopt = mean(f1_hist(end-50:end));
        if fopt < f0
            xopt = x1_hist(end,:);
            f0 = fopt;
        end
    end
end