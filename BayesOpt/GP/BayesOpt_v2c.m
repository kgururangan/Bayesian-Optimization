clear all
clc
close all

%p = genpath('/Users/karthikgururangan/Desktop/Karthik/Colormaps');
%addpath(p);

% needs row vectors!
fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C );

sigmaY = 0;

f = @(x) 2.5*exp(-(x(1)^2+x(2)^2)) + sin(2*x(1)) + cos(x(2));

xT = rand(1,2); yT = f(xT);

niter = 10;
kappa = 0.001;
paropt.nruns = 200; paropt.niter = 1e3; 
paropt.alpha = 0.01; paropt.beta = 1; 
paropt.normalize = 0;

%options = optimset('Display','Iter-Detailed');
options = [];
for i = 1:niter
    fprintf('Iteration-%d; FunEval = %4.2f\n',i,yT(end))
    
    mle = @(x) mlefun(xT,yT,x,sigmaY);
    jmle = @(x) jacmlefun(xT,yT,x,sigmaY);
    x0 = (3-0.05)*rand(1,size(xT,2)) + 0.05;
    [theta,fopt] = cg_optim_wrap(mle,jmle,x0,[],[],paropt.alpha,paropt.beta,paropt.nruns,paropt.niter,paropt.normalize);
    theta_hist(i,:) = theta;

    A = @(x) -acquisition_fcn(x,xT,yT,kappa,sigmaY,theta);
    x0 = rand(1,size(xT,2));
    [xsolve,fval] = fminsearch(A,x0,options);
    xT = [xT;xsolve];
    yT = [yT;f(xsolve)];
end

%% Plot convergence
hold on
for i = 2:size(xT,1)
    scatter( i, norm(xT(i,:)-xT(i-1,:)),80,'MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[0,0,0])
end
hold off
xlabel('Distance between consecutive x')
ylabel('y(x)')
grid on
set(gca,'FontSize',18,'Linewidth',2,'Box','off')

%%
N = 100;
P = N^2;
sx = linspace(min(xT(:,1)),max(xT(:,1)),100);
sy = linspace(min(xT(:,2)),max(xT(:,2)),100);
x = zeros(P,2); y = zeros(P,1);
count = 1;
for i = 1:N
    for j = 1:N
        y(count) = f([sx(i),sy(j)]);
        count = count + 1;
    end
end
y_rs = reshape(y,N,N);

figure(1)
surf(sx,sy,y_rs,'EdgeColor','none')
set(gca,'FontSize',18,'Linewidth',2,'Box','off')
colormap(fire)
hold on
for i = 1:niter
    scatter3(xT(i,1),xT(i,2),yT(i),100,'MarkerFaceColor',[0,0,0],'MarkerEdgeColor',[0,0,0])
    pause(0.5)
end
hold off

%% check and mle and jacmle are working properly
ind = randperm(size(x,1));  ii = ind(1:50);
xT = x(ii,:); yT = y(ii);
xTest = x(ind(51:end),:); yTest = y(ind(51:end));
mle = @(t) mlefun(xT,yT,t,sigmaY);
jmle = @(t) jacmlefun(xT,yT,t,sigmaY);
x0 = (3-0.05)*rand(1,size(xT,2)) + 0.05;
[theta,fopt] = cg_optim_wrap(mle,jmle,x0,[],[],paropt.alpha,paropt.beta,paropt.nruns,paropt.niter,paropt.normalize);
[Mu,Cov,~,~,~,~,~,~,~,~] = gprfcn(x,xT,yT,sigmaY,theta);
A = Mu + kappa*sqrt(diag(Cov));
%%
surfc(reshape(x(:,1),100,100),reshape(x(:,2),100,100),reshape(Mu,100,100),'EdgeColor','none')
hold on
surf(reshape(x(:,1),100,100),reshape(x(:,2),100,100),reshape(Mu+2*sqrt(diag(Cov)),100,100),'EdgeColor','none','FaceAlpha',0.5)
surf(reshape(x(:,1),100,100),reshape(x(:,2),100,100),reshape(Mu-2*sqrt(diag(Cov)),100,100),'EdgeColor','none','FaceAlpha',0.5)
scatter3(xT(:,1),xT(:,2),yT,50,'MarkerFaceColor',[0,0,0],'MarkerEdgeColor',[0,0,0])
scatter3(xTest(:,1),xTest(:,2),yTest,10,'MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[1,0,0])
hold off
%%
theta0 = linspace(1e-3,3,500);
F = zeros(length(theta0));
dF = zeros(length(theta0),length(theta0),2);
for i = 1:length(theta0)
    for j = 1:length(theta0)
        s = [theta0(i),theta0(j)];
        F(i,j) = mle(s);
        dF(i,j,:) = jmle(s);
    end
end
%%
m = 10;
[xx,yy] = meshgrid(theta0(1:m:end),theta0(1:m:end));
figure(1)
hold on
contour(theta0,theta0,F,80,'k-')
quiver(xx,yy,squeeze(dF(1:m:end,1:m:end,1)),squeeze(dF(1:m:end,1:m:end,2)),1,'r-')
axis square
colorbar
hold off
%% check that optim is working properly
x0 = rand(1,size(xT,2));
paropt.nruns = 10;
paropt.alpha = 0.01;
paropt.beta = 0.8;
paropt.niter = 1e3;
paropt.normalize = 0;
Xh = cell(1,paropt.nruns); Fh = cell(1,paropt.nruns);
for i = 1:paropt.nruns
    [x_hist,f_hist,jf_hist] = cg_optim(mle,jmle,x0,paropt.alpha,paropt.beta,paropt.niter,paropt.normalize)
    Xh{i} = x_hist; Fh{i} = f_hist;
end
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

function [] = plot_BO(GP_hist,theta_hist,it,sx,sy,xT,yT,xmax,ymax)
    
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
        cmap = fire();
        
        subplot(221)
        surfc(sx,sy,Mu_rs,'FaceAlpha',0.8); 
        colormap(cmap)
        hold on
        surf(sx,sy,Mu_rs+2*Sig_rs,'FaceAlpha',0.2)
        surf(sx,sy,Mu_rs-2*Sig_rs,'FaceAlpha',0.2)
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
