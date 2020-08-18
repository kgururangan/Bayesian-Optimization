clear all
clc
close all

% needs row vectors!
fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C );


% 1D
dim = 1;
P = 400;
x = linspace(-5,5,P)'; y = exp(-x.^2) + sin(2*x).*cos(2.*x);
nT = 15;
ind = randperm(P);
xTrain = x(ind(1:nT)); xTest = x(ind(nT+1:end));
yTrain = y(ind(1:nT)); yTest = x(ind(nT+1:end));

% 2D
% dim = 2;
% N = 50;
% P = N^2;
% s = linspace(-5,5,N);
% x = zeros(P,2); y = zeros(P,1);
% count = 1;
% for i = 1:N
%     for j = 1:N
%         x(count,:) = [s(i),s(j)];
%         y(count) = 2.5*exp(-(s(i)^2+s(j)^2)) + sin(3*s(i)) + cos(s(j));
%         count = count + 1;
%     end
% end
% y_rs = reshape(y,N,N);
% 
% nT = 50;
% ind = randperm(P);
% xTrain = x(ind(1:nT),:);  yTrain = y(ind(1:nT));
% xTest = x(ind(nT+1:end),:); yTest = y(ind(nT+1:end),:);

% data noise level (variance)
sigmaY = 0;

%% CG optimization
nruns = 100; niter = 1e3; alph = 0.01; beta = 1; theta1 = 0; normalize = 0;
mle = @(x) mlefun(xTrain,yTrain,x,sigmaY);
jmle = @(x) jacmlefun(xTrain,yTrain,x,sigmaY);
[xopt,fopt] = cg_optim_wrap(mle,jmle,x0,[],[],alph,beta,nruns,niter,normalize);
theta = xopt;
%% GP Regression
[Mu,Cov,logL,K,Ks,Kss,L,alph,v] =  gprfcn(x,xTrain,yTrain,theta,sigmaY);
Sig = sqrt(diag(Cov));

% PI Acquisition function
kappa = 1; tolsig = 1e-9;
Phi = @(x) cdf('norm',x,0,1);
phi = @(x) pdf('norm',x,0,1);
PI_acq = zeros(1,length(Mu));
i1 = find(Sig > tolsig);
Z = @(ind) (Mu(ind)-max(yTrain))./Sig(ind);
PI_acq(i1) = Phi(Z(i1)).*(Mu(i1)-max(yTrain))*kappa + phi(Z(i1)).*Sig(i1);

%% Plotting

if dim == 2
    Mu_rs = reshape(Mu,N,N); Sig_rs = reshape(sqrt(diag(Cov)),N,N);
    cmap = fire();
    subplot(121)
    surfc(s,s,Mu_rs,'FaceAlpha',0.8); 
    colormap(cmap)
    hold on;
    %surf(s,s,Mu_rs+2*Sig_rs,'FaceAlpha',0.1)
    %surf(s,s,Mu_rs-2*Sig_rs,'FaceAlpha',0.1)
    scatter3(xTrain(:,1),xTrain(:,2),yTrain,40,'MarkerFaceColor',[0,0,0])
    hold off
    axis square
    title('GP')
    set(gca,'FontSize',15,'Linewidth',2,'Box','off')
    subplot(122)
    surfc(s,s,y_rs)
    alpha 0.8
    axis square
    set(gca,'FontSize',15,'Linewidth',2,'Box','off')
    title('True')

else

    subplot(211)
    plot(x,y,'k-','Linewidth',2); hold on;
    plot(x,Mu,'k-','Linewidth',1,'color',[0.5,0,0]); 
    h = fill_between_lines(x', Mu'-2*sqrt(diag(Cov)'),Mu'+2*sqrt(diag(Cov)'),[0.3010, 0.7450, 0.9330]);
    h.FaceAlpha = 0.2; 
    scatter(xTrain,yTrain,'bo','MarkerFaceColor',[0,0,0.8])
    hold off
    xlabel('x')
    ylabel('y(x)')
    ll = legend('Data','GPR','\pm 2\sigma'); set(ll,'Location','NorthEast');
    set(gca,'FontSize',18,'Linewidth',2,'Box','off')
    title('Interpolant')
    grid on
    subplot(212)
    h = area(x,PI_acq); h.FaceAlpha = 0.2; h.FaceColor = [0,0.7,0];
    grid on
    set(gca,'FontSize',18,'Linewidth',2,'Box','off')
    xlabel('x')
    ylabel('Acq(x)')
    title('Acquisition Function')
    ll = legend('EI'); set(ll,'FontSize',18,'Location','NorthEast');
end


%% functions 
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
    L = sqrt(theta1)*L; Ks = theta1*Ks; Kss = theta1*Kss;
    alpha = L'\(L\yTrain);
    
    Mu = Ks*alpha;
    v = L\(Ks');
    Cov = Kss - v'*v; indneg = find(Cov<1e-60);  Cov(indneg) = 0;
    logL = -1/2*yTrain'*alpha - sum(log(diag(L))) - length(K)/2*log(2*pi);
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
    theta1 = 1/length(K)*yTrain'*(L'\(L\yTrain));
    L = sqrt(theta1)*L;
    alpha = L'\(L\yTrain);
    logL = -1/2*yTrain'*alpha - sum(log(diag(L))) - length(K)/2*log(2*pi);
    logL = -logL;
end

function [JlogL] = jacmlefun(xTrain,yTrain,theta,sigmaY)

    [K,Daa] = kernfcn(xTrain,xTrain,theta);
    if sigmaY == 0
        L = chol(K+1e-15*length(K)*eye(length(K)),'lower');
    else
        L = chol(K+sigmaY^2*eye(length(K)),'lower');
    end
    theta1 = 1/length(K)*yTrain'*(L'\(L\yTrain));
    L = sqrt(theta1)*L;
    alpha = L'\(L\yTrain); A = alpha*alpha';
    
    JlogL = zeros(1,length(theta));
    
    %JlogL(1) = theta(1)*trace( A*(L*L') - eye(length(K)) );
    for i = 1:length(theta)
        dKdthi = 1/theta(i)^3*Daa{i}.*(L*L');
        JlogL(i) = 1/2*trace( A*dKdthi - L'\(L\dKdthi) );
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

    [x1_hist,f1_hist,~] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize);
    [f0,i0] = min(f1_hist);
    xopt = x1_hist(i0,:);
    
    for i = 2:nruns
        if isempty(lb) && isempty(ub)
            x0 = rand(1,length(x0));
        else
            x0 = (ub-lb).*rand(1,length(x0)) + lb;
        end
        [x1_hist,f1_hist,~] = cg_optim(mle,jmle,x0,alpha,beta,niter,normalize);
        [fopt,iopt] = min(f1_hist);
        if fopt < f0
            xopt = x1_hist(iopt,:);
            f0 = fopt;
        end
    end
end