%% Generating Data from Lorenz System and recovering it
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
dt=0.05;
t_final=2.5; 
tspan=0:dt:t_final;

% Number of data points 
N_tilde = length(tspan);

% Run ODE solver to generate time series data
ODEoptions = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,D));
[~,x_clean]=ode89(@(t,x) lorenz(t,x,sigma,beta,rho),tspan,x0,ODEoptions);

%% Compute Derivative and Add noise
eps_x=0.05;
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
% int_pt=12;
% P=2;
% Q=2;
% [I,D1]=weak(N_tilde,int_pt,P,Q,dt);

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

%% Display ground truth
if polyorder == 2
    Xi_truth = zeros(10,3);
elseif polyorder == 3
    Xi_truth = zeros(20,3);
end    
% dy = [ sigma*(y(2)-y(1)) ; y(1)*(rho-y(3))-y(2) ; y(1)*y(2)-beta*y(3) ]
%               [   xdot , ydot , zdot  ]
Xi_truth(2,:) = [ -sigma ,  rho ,     0 ]; % y(1)
Xi_truth(3,:) = [ +sigma ,   -1 ,     0 ]; % y(2)
Xi_truth(4,:) = [      0 ,    0 , -beta ]; % y(3)
Xi_truth(6,:) = [      0 ,    0 ,     1 ]; % y(1) * y(2)
Xi_truth(7,:) = [      0 ,   -1 ,     0 ]; % y(1) * y(3)
disp('Ground Truth');
poolDataLIST({'x','y','z'},Xi_truth,D,polyorder,usesine);

%% Sparse regression: sequential threshold least squares (STLS, SINDy)
%  from Brunton, Proctor & Kutz (2016, PNAS)

% Thresholding hyperparameter, or sparsification knob.
lambda = 0.3;  

% SINDy !!
Xi_S = sparsifyDynamics(Theta,dx,lambda,D);

% Display parameter estimate
disp('From SINDy');
poolDataLIST({'x','y','z'},Xi_S,D,polyorder,usesine);

%% Sparse regression: Bayesian-SINDy (with Noise Iteration)
% Assuming the Priors have zero mean and variance of
PparamV=25^2;% Arbitrary large variance with zero mean for all coefficients
priorA=speye(size(Theta,2))/PparamV; % Inverse of covariance in the prior of param.

% Bayesian-SINDy !!
Xi_B=BayesianRegressGreedy_NoiseIter(Theta,dx,priorA,var_dx,Theta_Var);

% Display parameter estimate
disp('From Bayesian-SINDy');
poolDataLIST({'x','y','z'},Xi_B,D,polyorder,usesine);

return

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
poolDataLIST({'x','y','z'},Xi_R,D,polyorder,usesine);

return

%% Figure: Comparing prediction from Bayesian-SINDy with the Truth
tspan = [0 10];
[tA,xA]=ode89(@(t,x)lorenz(t,x,sigma,beta,rho),tspan,x0);   % true model
[tB,xB]=ode89(@(t,x)sparseGalerkin(t,x,Xi_B,polyorder,usesine),tspan,x0);  % approximate

% System Trajectory View
figure('Position',[100 100 600 300])
subplot(1,2,1)
plot3(xA(:,1),xA(:,2),xA(:,3),'LineWidth',1.5);
view(27,16)
grid on
xlabel('x','FontSize',13)
ylabel('y','FontSize',13)
zlabel('z','FontSize',13)

subplot(1,2,2)
plot3(xB(:,1),xB(:,2),xB(:,3),'LineWidth',1.5);
view(27,16)
grid on
xlabel('x','FontSize',13)
ylabel('y','FontSize',13)
zlabel('z','FontSize',13)

% Time evolution view
figure('Position',[100 100 900 300])
subplot(1,3,1)
plot(tA,xA(:,1),'k','LineWidth',1.5), hold on
plot(tB,xB(:,1),'r--','LineWidth',1.5)
grid on
xlabel('Time','FontSize',13)
ylabel('x','FontSize',13)

subplot(1,3,2)
plot(tA,xA(:,2),'k','LineWidth',1.5), hold on
plot(tB,xB(:,2),'r--','LineWidth',1.5)
grid on
xlabel('Time','FontSize',13)
ylabel('y','FontSize',13)

subplot(1,3,3)
plot(tA,xA(:,3),'k','LineWidth',1.5), hold on
plot(tB,xB(:,3),'r--','LineWidth',1.5)
grid on
xlabel('Time','FontSize',13)
ylabel('z','FontSize',13)
