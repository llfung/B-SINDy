%% Fig 9: Collinearity between x_1 and x_2 in dynamics of x_2 in Lorenz
%  Part of the paper "Rapid Bayesian identification of sparse 
%                     nonlinear dynamics from scarce and noisy data"
%       by L. Fung, U. Fasel, M. P. Juniper
% Copyright 2023, All Rights Reserved
% Code by Lloyd Fung and Matthew Juniper
% Based on code by Steven L. Brunton
%   For Paper, "Discovering Governing Equations from Data:
%            Sparse Identification of Nonlinear Dynamical Systems"
%   by S. L. Brunton, J. L. Proctor, and J. N. Kutz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialisation
clear all, close all, clc
figpath = './figs/';
addpath(genpath('./'));

% Set highest polynomial order of the combinations of polynomials of the state vector
polyorder = 3;
% Disable sin and cos of variables in the library (legacy)
usesine = 0;
% Set the parameters of the Lorenz system (chaotic)
sigma = 10;  
beta = 8/3;
rho = 28;
% Set the number of variables in the system
D = 3; % Lorenz has 3 dimensions

%% Generate Data
x0=[-1,6,15];  % Initial condition
% Time 
dt=0.025;
t_final=2.5; 
tspan=0:dt:t_final;

% Number of data points 
N_tilde = length(tspan);

% Run ODE solver to generate time series data
ODEoptions = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,D));
[t,x_clean]=ode89(@(t,x) lorenz(t,x,sigma,beta,rho),tspan,x0,ODEoptions);

%% Compute Derivative and Add noise
eps_x=0.2;
x = x_clean + eps_x*randn(size(x_clean));

%% Build library of nonlinear time series
% If each row of x contains [ x , y , z ] then each row of Theta contains (for polyorder=3):
% [1 , x , y , z , xx , xy , xz , yy , yz , zz , xxx , xxy , xxz , xyy , xyz , xzz , yyy , yyz , yzz , zzz ]

% Library of polynomials of state variables (x,y,z)
Theta_tilde = poolData(x,D,polyorder,usesine);
% Estimated Variance of the library of polynomials
%   assuming variables are independent
Theta_tildeVar = poolDataVar(x,eps_x^2*ones(size(x)),polyorder);
% Save the number of polynomial combinations
M = size(Theta_tilde,2);

%% Computing time derivatives from time series
%    assuming time series are regularly sampled (i.e. constant dt)

% Finite Differences
int_pt=12; % Finite Difference order (number of points used to compute derivative -1)
[I,D1]=FD(N_tilde,int_pt,dt);

% Apply derivatives
dx=D1*x;
Theta=I*Theta_tilde;

% Variance of noise in the library
Theta_Var = (I.^2)*Theta_tildeVar;
% Variance of noise in the time derivative
var_dx=(D1).^2*eps_x^2*ones(size(x));

%% Update number of points
disp(['Number of Data Points in time series: ' num2str(N_tilde)]);
N=N_tilde-int_pt;
disp(['Number of Data Points for regression: ' num2str(N)]);

%% Sparse regression: Bayesian-SINDy (with Noise Iteration)
% Assuming the Priors have zero mean and variance of
PparamV=25^2;% Arbitrary large variance with zero mean for all coefficients
priorA=speye(size(Theta,2))/PparamV; % Inverse of covariance in the prior of param.

% Bayesian-SINDy !!
Xi_B=BayesianRegressGreedy_NoiseIter(Theta,dx,priorA,var_dx,Theta_Var);

% Display parameter estimate
disp('From Bayesian-SINDy');
poolDataLIST({'x','y','z'},Xi_B,D,polyorder,usesine);

%% FIGURE 9:  Comparing dx_2/dt- rho x_1 + x_1+x_3 with x_1 and x_2
savefig=false;
ind=floor((int_pt+1)/2);
tPlot=t(ind+1:end-ind);

% Residue: dx_2/dt- rho x_1 + x_1+x_3 
res = dx(:,2)-rho*Theta(:,2)+Theta(:,7);

FigObj=figure('Position',[100 100 800 300]);
a=gca; hold on;
% Residue: dx_2/dt- rho x_1 + x_1+x_3 
p1=plot(tPlot,res,'x','MarkerSize',10);
% Best fit x_1
p2=plot(tPlot,(Theta(:,2)\res)*Theta(:,2),'LineWidth',2);
% Best fit x_2
p3=plot(tPlot,(Theta(:,3)\res)*Theta(:,3),'LineWidth',2);
% Zero
plot(tPlot,zeros(size(tPlot)),'k:');

% Labelling and make it pretty
a.FontSize=14;ylim([-45 45]);xlim([0 2.5]);
xlabel('$$t$$','Interpreter','latex','FontSize',14);
ylabel('Terms for $$\dot{x}_2$$','Interpreter','latex','FontSize',14);
% Legend
legend([p1 p2 p3],{'$$\dot{x}_2 - r x_1 + x_1 x_3$$','Scaled $$x_1$$','Scaled $$x_2$$'},...
    'Interpreter','latex','FontSize',12,'Location','northeast');
hold off; grid on;


if savefig
    saveas(FigObj,[figpath 'Lorenz_x2_res_eps' num2str(eps_x) '_dt' num2str(dt) '.fig']);
    saveas(FigObj,[figpath 'Lorenz_x2_res_eps' num2str(eps_x) '_dt' num2str(dt) '.svg'],'svg');
end