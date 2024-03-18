%% Fig 5: Example of learning in Van Der Pol system
%  Using Bayesian-SINDy and the original SINDy (STLS)
%
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
rng(498456);

% Set highest polynomial order of the combinations of polynomials of the state vector
polyorder = 3;
% Disable sin and cos of variables in the library (legacy)
usesine = 0;
% Set the parameters of the nonlinear oscillatory system (chaotic)
param.alpha = -0.1;  
param.beta  = -2.0;
param.gamma =  2.0;
param.delta = -0.1;
% Set the number of variables in the system
D = 2; % The system has 2 dimensions

%% Generate Data
x0=[1,0];  % Initial condition
% Time 
dt=0.05;
t_final=5; 
tspan=0:dt:t_final;

% Number of data points 
N_tilde = length(tspan);

% Run ODE solver to generate time series data
ODEoptions = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,D),'InitialStep',1e-6);
[t,x_clean]=ode89(@(t,x) nonlinearoscillator(t,x,param),tspan,x0,ODEoptions);

%% Compute Derivative and Add noise
eps_x=0.01; %(b) eps_x=0.005;
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

% Weak Formulation - Test Function phi(t)=(t^Q-1)^P
int_pt=6;
P=4;
Q=2;
[I,D1]=weak(N_tilde,int_pt,P,Q,dt);

% Finite Differences
% int_pt=6; % Finite Difference order (number of points used to compute derivative -1)
% [I,D1]=FD(N_tilde,int_pt,dt);

% Apply derivatives
dx=D1*x;
Theta=I*Theta_tilde;

% Variance of noise in the library
Theta_Var = (I.^2)*Theta_tildeVar;
% Variance of noise in the time derivative
var_dx=(D1).^2*eps_x^2*ones(size(x));

%% Sparse regression: sequential threshold least squares (STLS, SINDy)
%  from Brunton, Proctor & Kutz (2016, PNAS)

% Thresholding hyperparameter, or sparsification knob.
lambda = 0.06;  

% SINDy !!
Xi_S = sparsifyDynamics(Theta,dx,lambda,D);

% Display parameter estimate
disp('From SINDy');
poolDataLIST({'x','y'},Xi_S,D,polyorder,usesine);

%% Sparse regression: Bayesian-SINDy (with Noise Iteration)
% Assuming the Priors have zero mean and variance of
PparamV=1^2;% Arbitrary large variance with zero mean for all coefficients
priorA=speye(size(Theta,2))/PparamV; % Inverse of covariance in the prior of param.

% Bayesian-SINDy !!
Xi_B=BayesianRegressGreedy_NoiseIter(Theta,dx,priorA,var_dx,Theta_Var);

% Display parameter estimate
disp('From Bayesian-SINDy');
poolDataLIST({'x','y'},Xi_B,D,polyorder,usesine);

%% Sparse Bayes (RVM) (Tippings 2001, 2003)
OPTIONS		= SB2_UserOptions('iterations',10000,...
							  'diagnosticLevel', 0,...
							  'monitor', 10,...
                              'FixedNoise',false);
SETTINGS	= SB2_ParameterSettings();

% Initialise output
Xi_R = zeros(M,D);

% Perform regression using SparseBayes !!
for i=1:D
    % Now run the main SPARSEBAYES function
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
        SparseBayes('Gaussian', Theta, dx(:,i), OPTIONS, SETTINGS);
        Xi_R(PARAMETER.Relevant,i)	= PARAMETER.Value;
end

% Display parameter estimate
disp('From SparseBayes');
poolDataLIST({'x','y'},Xi_R,D,polyorder,usesine);

%% Preparing the data for Plotting figure 7
savefig = false;
% Evaluate approximated model from each method
tspan = [0 t_final];
[tA,xA]=ode89(@(t,x)nonlinearoscillator(t,x,param),tspan,x0,ODEoptions);   % true model
[tS,xS]=ode89(@(t,x)sparseGalerkin(t,x,Xi_S,polyorder,usesine),tspan,x0,ODEoptions);  % SINDY approximate
[tB,xB]=ode89(@(t,x)sparseGalerkin(t,x,Xi_B,polyorder,usesine),tspan,x0,ODEoptions);  % BINDY approximate
[tR,xR]=ode89(@(t,x)sparseGalerkin(t,x,Xi_R,polyorder,usesine),tspan,x0,ODEoptions);  % RVM approximate

%% Generating Figure 7
savefig = false;

FigObj2=figure('Position',[100 100 350 300]);
a=gca; hold on;
clr=colororder("gem");
plot(xA(:,1),xA(:,2),'k-','LineWidth',5);
plot(x(:,1),x(:,2),'kx','MarkerSize',12);
plot(xS(:,1),xS(:,2),'-','LineWidth',3,'Color',clr(1,:));
plot(xB(:,1),xB(:,2),'-','LineWidth',2,'Color',clr(2,:));
plot(xR(:,1),xR(:,2),'-','LineWidth',1.5,'Color',clr(3,:));

a.FontSize=16;
xlabel('$$x_1$$','Interpreter','latex','FontSize',20);
ylabel('$$x_2$$','Interpreter','latex','FontSize',20);
% legend('Truth','Noisy Data','SINDy','B-SINDy','SparseBayes','FontSize',14,'Location','best');
hold off;
axis([-1.1 1.1 -1.1 1.1]);
box on;grid on;
axis equal;
if savefig
    saveas(FigObj2,[figpath 'NonlinearOsc_x1x2_eps' num2str(eps_x) '_dt' num2str(dt) '.fig']);
    saveas(FigObj2,[figpath 'NonlinearOsc_x1x2_eps' num2str(eps_x) '_dt' num2str(dt) '.svg'],'svg');
end