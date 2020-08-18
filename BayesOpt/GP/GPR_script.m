clear all
clc
close all

addpath('/Users/karthik/Documents/MATLAB/2DPES/QPT');
addpath('/Users/karthik/Documents/MATLAB/Colormaps');
addpath('/Users/karthik/Documents/MATLAB/Additional_Functions');
%addpath('/Users/karthik/Documents/MATLAB/Additional_Functions/chebfun-master-2');
pltc = default_cmap();

fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C );

P = 400;
s = linspace(-5,5,P);
Y = exp(-s.^2) + sin(2*s).*cos(2.*s);

deg = 0;
%nsamp = 50;
%xT = s(1:nsamp:end)';
%yT = Y(1:nsamp:end)';
nT = 8;
ind = randperm(P);
xT = s(ind(1:nT))';
yT = Y(ind(1:nT))';

% MLE optimization to obtain hyperparameters
options = optimoptions('LSQNONLIN','Display','iter-detailed','Diagnostics','on','Algorithm','levenberg-marquardt');
options.FunctionTolerance = 1e-10;
options.OptimalityTolerance = 1e-12;
options.StepTolerance = 1e-14;

epsilon = 1e-6;
fun = @(theta) mle(theta,xT,yT,deg,epsilon);
nruns = 100;
theta_hist = zeros(nruns,2); fval_hist = zeros(1,nruns);
for i = 1:nruns
    lb = []; ub = [];
    x0 = rand(1,1);
    [xsolve,fval] = lsqnonlin(fun,x0,lb,ub,options);
    theta_hist(i,:) = xsolve; fval_hist(i) = fval;
end

% Build Kriging model
[~,I] = min(fval_hist); xsolve = theta_hist(I,:);
K = kernel(xsolve,xT,xT);
[beta,sigma] = calcBetaSigma(xT,yT,K,deg);
mu = zeros(P,1); mse = zeros(P,1);
for i = 1:length(s)
    [x1,x2] = krigingPred(s(i),xT,yT,beta,sigma,xsolve,K,deg);
    mu(i) = x1; mse(i) = x2;
end

% PI Acquisition function
kappa = 1;
Phi = @(x) cdf('norm',x,0,1);
phi = @(x) pdf('norm',x,0,1);
PI_acq = Phi( (mu - max(yT)*(1+kappa))./sqrt(abs(mse)));

% Plotting
subplot(211)
plot(s,mu,'Linewidth',2,'color',pltc(7,:)); hold on; 
plot(xT,yT,'ro','MarkerSize',10,'MarkerFaceColor',[1,0,0])
plot(s,Y,'k-','Linewidth',2); 
h = fill_between_lines(s,mu'-2*sqrt(mse'),mu'+2*sqrt(mse'),pltc(6,:));
h.FaceAlpha = 0.2; hold off;
grid on
set(gca,'FontSize',18,'Linewidth',2,'Box','off')
xlabel('x')
ylabel('y(x)')
title('Gaussian Process Interpolant')
ll = legend('Posterior Mean','Training Set','Data','\pm 2\sigma'); set(ll,'FontSize',18,'Location','Best');
subplot(212)
h = area(s,PI_acq); h.FaceAlpha = 0.2; h.FaceColor = [0,0.7,0];
grid on
set(gca,'FontSize',18,'Linewidth',2,'Box','off')
xlabel('x')
ylabel('y(x)')
title('Acquisition Function')
ll = legend('PI'); set(ll,'FontSize',18,'Location','NorthEast');




%% 2D Datasets
N = 50;
P = N^2;
s = linspace(-5,5,N);
x = zeros(P,2); y = zeros(P,1);
count = 1;
for i = 1:N
    for j = 1:N
        x(count,:) = [s(i),s(j)];
        y(count) = 2.5*exp(-(s(i)^2+s(j)^2)) + sin(2*s(i)) + cos(s(j));
        count = count + 1;
    end
end
y_rs = reshape(y,N,N);

deg = 0;
nT = 100;
ind = randperm(P);
xT = x(ind(1:nT),:);  yT = y(ind(1:nT));

% MLE optimization to obtain hyperparameters
options = optimoptions('LSQNONLIN','Display','iter-detailed','Diagnostics','on','Algorithm','levenberg-marquardt');
options.FunctionTolerance = 1e-10;
options.OptimalityTolerance = 1e-12;
options.StepTolerance = 1e-14;

epsilon = 1e-6;
fun = @(theta) mle(theta,xT,yT,deg,epsilon);
nruns = 50;
theta_hist = zeros(nruns,1); fval_hist = zeros(1,nruns);
for i = 1:nruns
    lb = []; ub = [];
    x0 = rand(1,1);
    [xsolve,fval] = lsqnonlin(fun,x0,lb,ub,options);
    theta_hist(i) = xsolve; fval_hist(i) = fval;
end

% Build Kriging predictor and mean-square error
[~,I] = min(fval_hist); xsolve = theta_hist(I,:);
K = kernel(xsolve,xT,xT);
[beta,sigma] = calcBetaSigma(xT,yT,K,deg);
ypred = @(s) krigingPred(s,xT,yT,beta,sigma,xsolve,K,deg);
M = 50;
s2 = linspace(-5,5,M);
count = 1;
for i = 1:M
    for j = 1:M
        [u,v] = ypred([s2(i),s2(j)]);
        mu(count) = u;
        mse(count) = v;
        count = count + 1;
    end
end

mu_rs = reshape(mu,M,M); mse_rs = reshape(mse,M,M);

% PI Acquisition function
kappa = 0.1;
Phi = @(x) cdf('norm',x,0,1);
phi = @(x) pdf('norm',x,0,1);
PI_acq = Phi( (mu - max(yT)*(1+kappa))./sqrt(abs(mse)) );
PI_acq_rs = reshape(PI_acq,M,M);

%
cmap = jet_white;
subplot(221)
%[a,b] = conturf(s2,s2,mu_rs);
%set(b,'EdgeColor','None');
surfc(s2,s2,mu_rs);
grid on
axis square
colorbar
title('GP Mean')
colormap(cmap)
subplot(222)
%[a,b] = contourf(s,s,y_rs);
%set(b,'EdgeColor','None');
surfc(s,s,y_rs);
grid on
colorbar
axis square
title('Actual')
subplot(223)
%[a,b] = contourf(s2,s2,PI_acq_rs);
%set(b,'EdgeColor','None');
surfc(s2,s2,PI_acq_rs);
grid on
colorbar
axis square
title('Acquisition Function')

%% Bayesian optimization
