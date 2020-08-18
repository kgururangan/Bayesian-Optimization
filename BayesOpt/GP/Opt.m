%{
Minimize y = f(x) =x.^4-x.^3-7*x.^2+2*x-15 subject to y < 10*x.^2-20 and -2< x < 3
Notes:
    The problem is simple so no need to re-parametrize, scale, ...
    fmincon is used for minimization.
    fmincon is a gradient-based optimization technique and requires an initial point.
    To ensure global optimality, optimization should be performed multiple times with distinct initial points.
%}
function [x_opt, f_opt, exit_flag, Opt_history] = Opt()
% intial points
x_int = [-2, -1, 0, 1.5, 5]';
% number of times the optimization problem is solved
Nopt = size(x_int, 1);
% creating a matlab structure of size Nopt to record the optimization process
Opt_history(Nopt).x = [];
Opt_history(Nopt).iter=[];
Opt_history(Nopt).fval=[];
% specifying the optimization options
options = optimoptions('fmincon','Display','notify','TolCon',1e-9,'TolFun',1e-9, 'OutputFcn', ...
                    @ outfun, 'MaxIter', 200, 'MaxFunEvals', 1000);
% intializa the variables
x_opt=zeros(Nopt,1);
f_opt = zeros(Nopt,1);
exit_flag = zeros(Nopt,1);
for i = 1: Nopt
    [x_opt(i), f_opt(i), exit_flag(i)] = fmincon(@(x) obj_fun(x), x_int(i), ...
            [], [], [], [], -2, 3, @(x) const(x), options);
end
figure
subplot(1, 2, 1)
x = -2:.1:3;
plot(x, x.^4-x.^3-7*x.^2+2*x-15, '--b', 'LineWidth', 2)
hold on
plot(x, 10*x.^2-20, '--r', 'linewidth', 2)
box on
axis([-2 3 -33 -10])
xlabel('x')
ylabel('y')
legend('y', '10x^2-20');
subplot(1, 2, 2)
hold on
colors = {'r', 'b', 'g', 'm', 'k'};
for i = 1: Nopt
    plot(Opt_history(i).iter, Opt_history(i).fval, '--o', 'LineWidth', 2, 'markeredgecolor', colors{i}, 'markerfacecolor', colors{i}, 'markersize', 15 - 2*i, 'color', colors{i});
end
legend('Initial point = -2', 'Initial point = -1', 'Initial point = 0', 'Initial point = 1.5', 'Initial point = 5')
box on
xlabel('Iteration')
ylabel('Objective Function')
title('Optimization History')
set(findobj(gcf,'type','axes'),'FontName','Cambria','FontSize',14,'FontWeight','Bold', 'LineWidth', 2);
set(gcf, 'color', 'w')

% The functions called above are defined below. Notice that they are defined within opt().

function F = obj_fun(x)
F = x.^4-x.^3-7*x.^2+2*x-15;
end

function [C, Ceq] = const(x)
Ceq = [];
C = (x.^4-x.^3-7*x.^2+2*x-15) - (10*x.^2-20);
end

function stop = outfun(x,optimValues,state)
     stop = false;
     switch state
         case 'iter'
           Opt_history(i).x = [Opt_history(i).x; x];
           Opt_history(i).iter=[Opt_history(i).iter,optimValues.iteration];
           Opt_history(i).fval=[Opt_history(i).fval, optimValues.fval];
         otherwise
     end
end

end