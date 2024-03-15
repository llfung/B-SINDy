%% Recovering the Lotka-Volterra equations from Lynx-Hare Pelts Trade Data
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

%% Load Data
load('./data/LynxHare.mat','LynxHare');

%% Extracting meta-data
x=LynxHare(:,[2,1]);
N_tilde=size(LynxHare,1); % No. of DataPts
D=size(LynxHare,2); % No. of Dimensions

eps_x=2.7; %sigma_x=0.05; The  original value for precision, but process noise seems to suggest noise is much higher

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
% int_pt=8;
% P=2;
% Q=2;
% [I,D1]=weak(N_tilde,int_pt,P,Q,1);

% Finite Differences
int_pt=8; % Finite Difference order (number of points used to compute derivative -1)
[I,D1]=FD(N_tilde,int_pt,1);

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

%% Sparse regression: sequential threshold least squares (STLS, SINDy)
%  from Brunton, Proctor & Kutz (2016, PNAS)

% Thresholding hyperparameter, or sparsification knob.
lambda = 0.025;  

% SINDy !!
Xi_S = sparsifyDynamics(Theta,dx,lambda,D);

% Display parameter estimate
disp('From SINDy');
poolDataLIST({'x','y'},Xi_S,D,polyorder,usesine);

%% Sparse regression: Bayesian-SINDy (with Noise Iteration)
% Assuming the Priors have zero mean and variance of
PparamV=10^2;% Arbitrary large variance with zero mean for all coefficients
priorA=speye(size(Theta,2))/PparamV; % Inverse of covariance in the prior of param.

% Bayesian-SINDy !!
Xi_B=BayesianRegressGreedy_NoiseIter(Theta,dx,priorA,var_dx,Theta_Var);

% Display parameter estimate
disp('From Bayesian-SINDy');
poolDataLIST({'x','y'},Xi_B,D,polyorder,usesine);

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
poolDataLIST({'x','y'},Xi_R,D,polyorder,usesine);

return

%% COMPARE learnt result with "Correct" result and Data
LearnFunc=@(x,Xi)(([1 x(1) x(2) x(1)^2 x(1)*x(2) x(2)^2 x(1)^3 x(1)^2*x(2) x(1)*x(2)^2 x(2)^3]*Xi)');

[tB,xB]=ode45(@(t,x)sparseGalerkin(t,x,Xi_B,polyorder,usesine),[0 20],x(1,:)); % Approximate from BINDy
[tS,xS]=ode45(@(t,x)sparseGalerkin(t,x,Xi_S,polyorder,usesine),[0 20],x(1,:)); % Approxiimate from SINDy

figure;
plot([0:20]'+1900,x(:,1),'bx','MarkerSize',12);
hold on;
plot([0:20]'+1900,x(:,2),'rx','MarkerSize',12);
plot(tB+1900,xB(:,1),'b--');
plot(tB+1900,xB(:,2),'r--');
plot(tS+1900,xS(:,1),'b:');
plot(tS+1900,xS(:,2),'r:');
hold off;
legend('x_1 (data)','x_2 (data)',...
    'x_1 (B-SINDy)','x_2 (B-SINDy)',...
    'x_1 (SINDy)','x_2 (SINDy)')
xlabel('Time (year)');
ylabel('x_1: Hare / x_2: Lynx');
ylim([0 120]);
