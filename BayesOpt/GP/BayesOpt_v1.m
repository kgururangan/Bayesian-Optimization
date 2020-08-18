clear all
clc
close all

% needs row vectors!
fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C );

sigmaY = 0;
flag_plot = 1;

P = 400;
x = linspace(-5,5,P)'; 
y = exp(-x.^2) + sin(2*x).*cos(10.*x);
[ymax,imax] = max(y); xmax = x(imax);

ii = randi(P,1); xT = x(ii); yT = y(ii);

%
nruns = 10; niter = 5e3; alph = 0.01; beta = 1; normalize = 2;
mle = @(x) mlefun(xT,yT,x,sigmaY);
jmle = @(x) jacmlefun(xT,yT,x,sigmaY);
x0 = rand(1,size(xT,2));
[xopt,fopt] = cg_optim_wrap(mle,jmle,x0,[],[],alph,beta,nruns,niter,normalize);
theta = xopt;
theta_hist(1,:) = theta;

%
kappa = 0.01;
Phi = @(x) cdf('Normal',x,0,1);
phi = @(x) pdf('Normal',x,0,1);

%
maxit = 1000; 
flag = 0; 
tol = 1e-8; tolsig = 0;
it = 1;

theta_hist(it) = xopt;

if flag_plot == 1
    close all
    figure(1)
    set(gcf, 'Position',  [100, 100, 1000, 1000])
end

while flag == 0 && it <= maxit

    mess = sprintf('Iteration-%d',it); disp(mess)
    
    [Mu,Cov,~,~,~,~,~,~,~] =  gprfcn(x,xT,yT,theta,sigmaY);
    Sigsq = diag(Cov);
    Sig = sqrt(Sigsq);

    % PI Acquisition function
    PI_acq = zeros(1,length(Mu));
    i1 = find(Sig > tolsig);
    
    % option 1 - is it Sigsq or Sig?
    %Z = @(ind) (Mu(ind)-max(yT))./Sig(ind);
    %PI_acq(i1) = Phi(Z(i1)).*(Mu(i1)-max(yT))*kappa + phi(Z(i1)).*Sig(i1);
    
    % option 2
    imp = @(ind) Mu(ind)-max(Mu(ind))-kappa;
    Z = @(ind) imp(ind)./Sig(ind);
    PI_acq(i1) = Phi(Z(i1)).*imp(i1) + phi(Z(i1)).*Sig(i1);
    
    GP_hist{it,1} = Mu; GP_hist{it,2} = Cov; GP_hist{it,3} = PI_acq;
    
    if flag_plot == 1
            subplot(2,3,1:2)
            plot(x,y,'k-','Linewidth',2); hold on;
            scatter(xmax,ymax,140,'kp','MarkerFaceColor',[0,1,0])
            plot(x,Mu,'k-','Linewidth',1,'color',[0.5,0,0]); 
            h = fill_between_lines(x',Mu'-2*Sig',Mu'+2*Sig',[0.3010, 0.7450, 0.9330]);
            h.FaceAlpha = 0.2; 
            scatter(xT(1:it),yT(1:it),80,'b+','MarkerFaceColor',[0,0,0.8])
            hold off
            xlabel('x')
            ylabel('GP(x)')
            %ll = legend('Data','GPR','\pm 2\sigma'); set(ll,'Location','NorthEast');
            set(gca,'FontSize',18,'Linewidth',2,'Box','off')
            title(sprintf('Iteration - %d',it));
            grid on
            axis([-inf,inf,-2,3])
            subplot(2,3,4:5)
            h = area(x,PI_acq); h.FaceAlpha = 0.2; h.FaceColor = [0,0.7,0];
            grid on
            set(gca,'FontSize',18,'Linewidth',2,'Box','off')
            xlabel('x')
            ylabel('Acq(x)')
            %title('Acquisition Function')
            ll = legend('EI'); set(ll,'FontSize',18,'Location','NorthEast');
            %axis([-inf,inf,0,0.2])

            subplot(2,3,3)
            yh(it) = abs(max(yT)-ymax);
            yhh(it) = abs(max(yT)-max(Mu));
            plot(1:it,yh(1:it),'r-.','Linewidth',2); hold on
            hh1 = scatter(it,abs(max(yT)-ymax),140,'MarkerFaceColor',[1,0,0]); 
            plot(1:it,yhh(1:it),'b-.','Linewidth',2); 
            hh2 = scatter(it,abs(max(yT)-max(Mu)),140,'MarkerFaceColor',[0,0,1]); hold off
            xlabel('Iteration')
            ylabel('|y_{max}^{pred} - y_{max}|')
            axis([-inf,inf,0,max(abs(max(yT)-ymax))+1])
            ll = legend([hh1,hh2],['True Error','\mu Error'])
            set(gca,'FontSize',18,'Linewidth',2,'Box','off')
            grid on
            axis square
            
            subplot(2,3,6)
            plot(1:it,theta_hist,'k-.','Linewidth',2); hold on
            scatter(it,theta_hist(it),140,'MarkerFaceColor',[1,0,0]); hold off
            xlabel('Iteration')
            ylabel('\theta')
            axis([-inf,inf,0,2])
            set(gca,'FontSize',18,'Linewidth',2,'Box','off')
            grid on
            axis square
            pause(0.25)
    end
    
    [MPI,jj] = max(PI_acq);
    
    if MPI < tol
        flag = 1;
        mess = sprintf('Optimum found at Iter-%d',it);
        disp(mess)
        break;
    end
    
    xT = [xT;x(jj)];
    yT = [yT;y(jj)];
    
    it = it + 1;
    
    mle = @(x) mlefun(xT,yT,x,sigmaY);
    jmle = @(x) jacmlefun(xT,yT,x,sigmaY);
    x0 = theta;
    [xopt,fopt] = cg_optim_wrap(mle,jmle,x0,[],[],alph,beta,nruns,niter,normalize);
    theta = xopt;
    theta_hist(it) = xopt;
    
    
end

%% v2 

clear all
clc
close all

% data noise parameter
sigmaY = 0;

% objective function to be maximized
y = @(x) exp(-x.^2) + sin(2*x).*cos(10.*x);

% random first data point
lb = -5; ub = 5;
xT = (ub-lb).*rand(1,1) + lb;
yT = y(xT);

% kernel optimization parameters
nruns = 50; maxit = 1e3; alph = 0.01; beta = 1.0; normalize = 2;

% bayes opt parameters
niter = 30; 
kappa = [1];

yh = zeros(1,niter);

% bayes opt loop
for i = 1:niter
    
    mle = @(x) mlefun(xT,yT,x,sigmaY);
    jmle = @(x) jacmlefun(xT,yT,x,sigmaY);
    th0 = rand(1,size(xT,2));
    
    [xopt,fopt] = cg_optim_wrap(mle,jmle,th0,[],[],alph,beta,nruns,maxit,normalize);
    theta = xopt;
    theta_hist(i,:) = theta;
    
    for k = 1:length(kappa)
        A = @(x) -acquisition_fcn_ucb(x,kappa(k),xT,yT,theta,sigmaY);
        %A = @(x) -acquisition_fcn_ei(x,kappa,xT,yT,theta,sigmaY);

        options = optimset('TolFun',1e-8,'TolX',1e-8);
        x0 = rand(1,size(xT,2));
        [xsolve,fval] = fminsearch(A,x0,options); 

        xT = [xT; xsolve];
        yT = [yT; y(xsolve)];

        mess = sprintf('Iter-%d; Acq = %4.2f; y = %4.2f',i,fval,yT(end));
        disp(mess)
    end
    
end

[ybo,ibo] = max(yT);
xbo = xT(ibo,:);

%%
P = 400;
xp = linspace(-5,5,P); 
yp = y(xp);
[ymax,imax] = max(yp); xmax = xp(imax);

% needs row vectors!
fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C );

close all
figure(1)
set(gcf, 'Position',  [100, 100, 1000, 1000])

th = linspace(0,pi/2,size(xT,1))';
clr = [cos(th),sin(th),zeros(size(xT,1),1)];

for i = 1:size(xT,1)
    subplot(211)
    plot(xp,yp,'color',[0    0.4470    0.7410],'Linewidth',2); hold on
    for j = 1:i
        scatter(xT(j),yT(j),100,'MarkerFaceColor',clr(j,:),'MarkerEdgeColor',[0,0,0])
    end
    plot(xT(1:i),yT(1:i),'k-.','Linewidth',2)
    xlabel('x')
    ylabel('Objective Function f(x)')
    set(gca,'FontSize',18,'Linewidth',2,'Box','off')
    grid on
    
    subplot(212)
    plot(1:i,abs(yT(1:i)-ymax),'k-.','Linewidth',2); hold on
    h = scatter(1:i,abs(yT(1:i)-ymax),80,'MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[0,0,0]);
    hold off
    axis([1,size(xT,1),0,5])
    xlabel('Iteration')
    ylabel('Distance from Optimum')
    ll = legend(h,'Absolute Error'); set(ll,'Location','NorthEast');
    set(gca,'FontSize',18,'Linewidth',2,'Box','off')
    grid on
    
    pause(0.5)
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