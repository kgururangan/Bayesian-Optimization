clear all
clc
close all

% 1. load in concentration data 
load red3yellow3_4.15.19_processed.mat

addpath('/Users/karthik/Documents/MATLAB/Colormaps');

IMcc = IMbase;

p = 80;
X1 = IMcc(p,:);

N = size(IMcc,1);
x = zeros(size(IMcc,1),2); y = zeros(size(IMcc,1),1);
for i = 1:size(IMcc,1)
    y(i) = -mean( (IMcc(i,:)-X1).^2 );
    x(i,:) = Cc(:,i);
end
%
ff = 1e5;
x = x*ff;

%% BayesOpt 

% random first data point
nT = 1;
ind = randperm(size(IMcc,1)); ind(ind==p) = [];
xT = x(ind(1:nT),:); yT = y(ind(1:nT));
xTest = x(p,:); yTest = y(p);

% kernel optimization parameters
nruns = 100; maxit = 1e3; alph = 0.01; beta = 1.0; normalize = 2;

% bayes opt parameters
niter = 1000; 
kappa = 0.1;

% bayes opt loop
for i = 1:10
    
    mle = @(x) mlefun(xT,yT,x,sigmaY);
    jmle = @(x) jacmlefun(xT,yT,x,sigmaY);
    th0 = rand(1,size(xT,2));
    
    [xopt,fopt] = cg_optim_wrap(mle,jmle,th0,[],[],alph,beta,nruns,maxit,normalize);
    theta = xopt;
    theta_hist(i,:) = theta;
    
    A = @(x) -acquisition_fcn_ucb(x,kappa,xT,yT,theta,sigmaY);
    %A = @(x) -acquisition_fcn_ei(x,kappa,xT,yT,theta,sigmaY);
    
    options = optimset('Display','Iter-Detailed','TolFun',1e-8,'TolX',1e-8);
    x0 = rand(1,size(xT,2));
    [xsolve,fval] = fminsearch(A,x0,options); 
    
    xT = [xT; xsolve];
    yT = [yT; y(xsolve)];
    
    
end
%% functions

function [A] = acquisition_fcn_ucb(x,kappa,xTrain,yTrain,theta,sigmaY)
    [Mu,Cov,~,~,~,~,~,~,~] = gprfcn(x,xTrain,yTrain,theta,sigmaY);
    A = Mu + kappa*sqrt(diag(Cov));
end

function [A] = acquisition_fcn_ei(x,kappa,xTrain,yTrain,theta,sigmaY)
    Phi = @(x) cdf('norm',x,0,1);
    phi = @(x) pdf('norm',x,0,1);
    [Mu,Cov,~,~,~,~,~,~,~] = gprfcn(x,xTrain,yTrain,theta,sigmaY);
    Sig = diag(sqrt(Cov));
    A = zeros(1,length(Mu));
    i1 = find(Sig > 0);
    Z = @(ind) (Mu(ind) - max(Mu) - kappa)./Sig(ind);
    A(i1) = (Phi(Z(i1)).*Z(i1) + phi(Z(i1))).*Sig(i1);
end


function [Mu,Cov,logL,K,Ks,Kss,L,alpha,v] = gprfcn(xTest,xTrain,yTrain,theta,sigmaY)
    [K,~] = kernfcn(xTrain,xTrain,theta);
    [Ks,~] = kernfcn(xTest,xTrain,theta);
    [Kss,~] = kernfcn(xTest,xTest,theta);
    
    if sigmaY == 0
        L = chol(K+eps*length(K)*eye(length(K)),'lower');
    else
        L = chol(K+sigmaY^2*eye(length(K)),'lower');
    end
    
    theta1 = 1/length(K)*yTrain'*(L'\(L\yTrain));
    K = theta1*K; Ks = theta1*Ks; Kss = theta1*Kss;
    
    if sigmaY == 0
        L = chol(K+eps*length(K)*eye(length(K)),'lower');
    else
        L = chol(K+sigmaY^2*eye(length(K)),'lower');
    end
    
    alpha = L'\(L\yTrain);
    Mu = Ks*alpha;
    v = L\(Ks');
    Cov = Kss - v'*v; Cov(Cov<1e-60) = 0;
    %logL = -1/2*yTrain'*alpha - sum(log(diag(L))) - length(K)/2*log(2*pi);
    logL = -length(K)/2*(log(2*pi)+1) - length(K)/2*log( yTrain'*alpha / length(K) ) - sum(log(diag(L)));
    logL = -logL;

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
%     K = K*theta1;
%     if sigmaY == 0
%         L = chol(K+1e-15*length(K)*eye(length(K)),'lower');
%     else
%         L = chol(K+sigmaY^2*eye(length(K)),'lower');
%     end
    %logL = -1/2*yTrain'*alpha - sum(log(diag(L))) - length(K)/2*log(2*pi);
    logL = -length(K)/2*(log(2*pi)+1) - length(K)/2*log(theta1) - sum(log(diag(L)));
    logL = -logL;
end

function [JlogL] = jacmlefun(xTrain,yTrain,theta,sigmaY)

    [K,Daa] = kernfcn(xTrain,xTrain,theta);
    if sigmaY == 0
        L = chol(K+1e-15*length(K)*eye(length(K)),'lower');
    else
        L = chol(K+sigmaY^2*eye(length(K)),'lower');
    end
%    theta1 = 1/length(K)*yTrain'*(L'\(L\yTrain));
%     K = theta1*K;
%     if sigmaY == 0
%         L = chol(K+1e-15*length(K)*eye(length(K)),'lower');
%     else
%         L = chol(K+sigmaY^2*eye(length(K)),'lower');
%     end
    alpha = L'\(L\yTrain); 
    %A = alpha*alpha';
    JlogL = zeros(1,length(theta));
    for i = 1:length(theta)
        dKdthi = 1/theta(i)^3*Daa{i}.*(L*L');
        %JlogL(i) = 1/2*trace( A*dKdthi - L'\(L\dKdthi) );
        %JlogL(i) = 1/2*trace( length(K)*A*dKdthi - L'\(L\dKdthi) )/(yTrain'*alpha);
        JlogL(i) = 1/2*(length(K)*alpha'*dKdthi*alpha/(yTrain'*alpha) - trace(L'\(L\(dKdthi))) );
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

function [x_hist,f_hist,jf_hist,exitflag] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize)
    %mle = @(x) mlefun(xTrain,yTrain,[1,x],sigmaY);
    %jmle = @(x) jacmlefun(xTrain,yTrain,[1,x],sigmaY);
    x_hist = zeros(niter,length(x0)); x_hist(1,:) = x0;
    f_hist = zeros(niter,1); f_hist(1) = mle(x0);
    jf_hist = zeros(niter,length(x0)); jf_hist(1,:) = jmle(x0);
    dtemp = jf_hist(1,:);
    k = 2;
    exitflag = 0;
    while  k <= niter 
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
        if norm(x_hist(k,:) - x_hist(k-1,:)) <= 1e-9
            exitflag = 1;
        end
        k = k + 1;
        %alpha = alpha/k;
    end
end

function [xopt,fopt] = cg_optim_wrap(mle,jmle,x0,lb,ub,alpha,beta,nruns,niter,normalize)

    mm = 50;
    
    [x1_hist,f1_hist,~] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize);
    %[fopt,iopt] = min(f1_hist);
    fopt = mean(f1_hist(niter-mm:end,:));
    xopt = x1_hist(end,:);
    %xopt = x1_hist(iopt,:);
    %fopt = f1_hist(end);
    %xopt = x1_hist(end,:);
    
    for i = 2:nruns
        if isempty(lb) && isempty(ub)
            x0 = rand(1,length(x0));
        else
            x0 = (ub-lb).*rand(1,length(x0)) + lb;
        end
        [x1_hist,f1_hist,~,~] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize);
        %[f0,i0] = min(f1_hist);
        f0 = mean(f1_hist(niter-mm:end,:));
        %f0 = f1_hist(end);
        if fopt > f0
            xopt = x1_hist(end,:);
            %xopt = x1_hist(i0,:);
            fopt = f0;
        end
    end
end

function [] = plot_BO(GP_hist,theta_hist,it,sx,sy,xT,yT,xmax,ymax)

    cmap = fire();
    
    max_mu = max(GP_hist{1,1}); max_pi = max(GP_hist{1,3});
    min_mu = min(GP_hist{1,1}); min_pi = min(GP_hist{1,3});
    for i = 2:it
        temp1 = max(GP_hist{i,1});
        temp2 = max(GP_hist{i,3});
        temp3 = min(GP_hist{i,1});
        temp4 = min(GP_hist{i,3});
        if max_mu < temp1
            max_mu = temp1;
        end
        if max_pi < temp2
            max_pi = temp2;
        end
        if min_mu > temp3
            min_mu = temp3;
        end
        if min_pi > temp4
            min_pi = temp4;
        end
    end
        
    clf
    figure(1)
    set(gcf, 'Position',  [100, 100, 1000, 1000])
    
    Nx =length(sx); Ny = length(sy);
    for k = 1:it
        Mu = GP_hist{k,1}; Cov = GP_hist{k,2}; PI_acq = GP_hist{k,3};
        Mu_rs = reshape(Mu,Ny,Nx); Sig_rs = reshape(sqrt(diag(Cov)),Ny,Nx); PI_rs = reshape(PI_acq,Ny,Nx);
        
        subplot(221)
        surfc(sx,sy,Mu_rs,'FaceAlpha',0.8); 
        colormap(cmap)
        hold on
        surf(sx,sy,Mu_rs+2*Sig_rs,'FaceAlpha',0.3,'EdgeColor','none')
        surf(sx,sy,Mu_rs-2*Sig_rs,'FaceAlpha',0.3,'EdgeColor','none')
        scatter3(xT(1:k,1),xT(1:k,2),yT(1:k),100,'MarkerFaceColor',[0,0,0.9])
        scatter3(xmax(1),xmax(2),ymax,120,'kp','MarkerFaceColor',[0,1,0])
        hold off
        %axis square
        xlabel('x')
        ylabel('y')
        zlabel('GP(x,y)')
        title(sprintf('Iteration - %d',k))
        set(gca,'FontSize',15,'Linewidth',2,'Box','off')
        axis([-inf,inf,-inf,inf,min_mu-0.1,max_mu+0.1])
        
        subplot(222)
        surfc(sx,sy,PI_rs)
        xlabel('x')
        ylabel('y')
        zlabel('Acq(x,y)')
        title('Expected Improvement')
        %axis square
        set(gca,'FontSize',15,'Linewidth',2,'Box','off')
        axis([-inf,inf,-inf,inf,min_pi-0.1,max_pi+0.1])
        
        subplot(223)
        plot(1:it,abs(yT(1:it)-ymax),'k-.','Linewidth',2); hold on
        H = scatter(k,abs(yT(k)-ymax),100,'MarkerFaceColor',[1,0,0]); 
        xlabel('Iteration')
        ylabel('|y_{max}^{pred} - y_{max}|')
        ll = legend(H,'Error'); set(ll,'Location','NorthWest');
        axis([-inf,inf,0,max(abs(yT-ymax))+1])
        set(gca,'FontSize',18,'Linewidth',2,'Box','off')
        grid on
        axis square

        subplot(224)
        clr = [1,0,0;0,0,1];
        hold on
        for t = 1:size(theta_hist,2)
            plot(1:it,theta_hist(1:it,t),'k:','color',clr(t,:),'Linewidth',2); 
            h{t} = scatter(k,theta_hist(k,t),100,'MarkerFaceColor',clr(t,:),'MarkerEdgeColor',[0,0,0]); 
        end
        hold off
        ll = legend([h{1},h{2}],'\theta_1','\theta_2'); set(ll,'Location','NorthWest');
        xlabel('Iteration')
        ylabel('\theta')
        axis([-inf,inf,min(min(theta_hist))-0.1,max(max(theta_hist))+0.1])
        set(gca,'FontSize',18,'Linewidth',2,'Box','off')
        grid on
        axis square

        pause(0.5)
    end
        
end
