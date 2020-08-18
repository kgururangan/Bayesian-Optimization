clear all
clc
close all

% fit to real part of data, imag introduces instability for now

t = linspace(0,1,200)';
%y = zeros(length(t),1);

% K = 5;
% omega = 300*rand(1,K); gamma = -10*rand(1,K); c = 4*rand(1,K)-2;
% for i = 1:K
%     y = y + c(i)*exp(1i*omega(i)*t+gamma(i)*t);
% end
y = 2*exp(-10*t) - 2*exp(-20*t);

y = real(y);

verbose = 1;

nparams = 4;
%fcn = @(x) x(1)*exp(1i*x(2)*t+x(3)*t) + x(4)*exp(1i*x(5)*t+x(6)*t);
fcn = @(x) x(1)*exp(x(2)*t) + x(3)*exp(x(4)*t);

model = @(x) real(fcn(x));
f = @(x) (model(x)-y).^2;


%
lambd = 0.01;
ntrials = 10;
lambda_t = kron(lambd,ones(1,ntrials));
niter = 300;

x_hist = cell(size(lambda_t));
loss_hist = zeros(size(lambda_t));

% 
tic
% loop over values of lambda
for a1 = 1:size(lambda_t,1)
    for a2 = 1:size(lambda_t,2)
        
        lambda = lambda_t(a1,a2);
        x0 = rand(nparams,1);
        x = x0;
        r0 = mean(f(x0));
        
        % LM iteration loop
        for i = 2:niter
            % parameter increment
            [delta,~,~,~] = levmarstep(f,x,lambda,[]);
            % calculate residual
            r = mean(f(x+delta));
            % print things
            if verbose == 1
                fprintf('iter- %d    Loss = %4.2f    lambda = %4.2f\n',i,r,lambda);
            end
            % update design and damping parameters
            if r < r0
                r0 = r;
                lambda = lambda/10;
                x = x + delta;
            else
                lambda = lambda*10;
            end
        end
        
        % record loss and solution
        loss_hist(a1,a2) = r;
        x_hist{a1,a2} = x;
        
    end
end
toc

%%
[~,ind] = min(loss_hist(:));
[i1,i2] = ind2sub(size(loss_hist),ind);
xout = x_hist{i1,i2};

fout = model(xout);

figure(1)
plot(t,y,'ko','MarkerSize',5,'MarkerFaceColor',[0.1,0.1,0.1])
hold on
plot(t,fout,'k-.','color',[0,0,1],'Linewidth',2)
hold off
set(gca,'FontSize',17,'Linewidth',2,'Box','off')
grid on
ll= legend('Data','LM Fit'); set(ll,'FontSize',15,'Location','NorthEast');

%%
function [J,F] = jacobian(f,x)
    F = f(x);
    I = eye(length(x));
    dx = 0.1;
    %if length(size(F)) ~= 1
    %    J = reshape(kron(zeros(length(x),1),zeros(size(F'))),size(F,1),size(F,2),length(x));
    %else
    %    J = zeros(length(F),length(x));
    %end
    J = zeros(numel(F),length(x));
    for i = 1:length(x)
        F1 = f(x+dx*I(:,i));
        F2 = f(x-dx*I(:,i));
        F3 = f(x+2*dx*I(:,i));
        F4 = f(x-2*dx*I(:,i));
        Temp = 1/(12*dx)*(-F3+8*F1-8*F2+F4);
        J(:,i) = Temp(:);
        %J(:,i) = 1/(12*dx)*(-F3+8*F1-8*F2+F4);
        %J(:,i) = 1/(2*dx)*(F1-F2);
    end
end

function [J,H,F] = jachess(f,x)
    F = f(x);
    I = eye(length(x));
    dx = 0.1;
    J = zeros(numel(F),length(x));
    H = zeros(numel(F),length(x),length(x));
    for i = 1:length(x)
        F1 = f(x+dx*I(:,i));
        Temp1 = 1/dx*(F1-F);
        J(:,i) = Temp1(:);
        for j = 1:length(x)
            F2 = f(x+dx*I(:,i)+dx*I(:,j));
            if j ~= i
                F3 = f(x+dx*I(:,j));
            else
                F3 = F1;
            end
            Temp2 = 1/dx^2*(F2 - F1 - F3 + F);
            H(:,i,j) = Temp2(:);
        end
    end
end

function [delta,F,A,b] = levmarstep(f,x,lambda,J0)
    %if size(y,1) < size(y,2)
    %    y = y';
    %end
    [J,F] = jacobian(f,x);
    JJ = J'*J;
    if isempty(J0)
        A = JJ + lambda*eye(length(x));
    else
        A = JJ + lambda*diag(J0'*J0);
    end
    b = -J'*F;
    %delta = A\b;
    L = chol(A,'lower');
    delta = L'\(L\b);
end
