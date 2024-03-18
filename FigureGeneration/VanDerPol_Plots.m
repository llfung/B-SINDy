%% Fig 3: Example of learning in Van Der Pol system
%    Comparison between SINDy (STLS), B-SINDy and SparseBayes
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
rng(18);

% Set highest polynomial order of the combinations of polynomials of the state vector
polyorder = 3;
% Disable sin and cos of variables in the library (legacy)
usesine = 0;
% Set the parameters of the Van Der Pol system (chaotic)
param.beta  = 4;
% Set the number of variables in the system
D = 2; % Van Der Pol has 2 dimensions

%% Generate Data
x0=[2,0];  % Initial condition
% Time 
dt=0.15;
t_final=12; 
tspan=0:dt:t_final;

% Number of data points 
N_tilde = length(tspan);

% Run ODE solver to generate time series data
ODEoptions = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,D),'InitialStep',1e-6);
[t,x_clean]=ode15s(@(t,x) vanderpol(t,x,param),tspan,x0,ODEoptions);

%% Compute Derivative and Add noise
eps_x=0.1;
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
% int_pt=8;
% P=2;
% Q=2;
% [I,D1]=weak(N_tilde,int_pt,P,Q,dt);

% Finite Differences
int_pt=8; % Finite Difference order (number of points used to compute derivative -1)
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

%% Sparse regression: sequential threshold least squares (STLS, SINDy)
%  from Brunton, Proctor & Kutz (2016, PNAS)

% Thresholding hyperparameter, or sparsification knob.
lambda = 0.4;  

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

%% Preparing the data for Plotting figure 3
savefig = false;
tspan = [0 t_final];
[tA,xA]=ode15s(@(t,x)vanderpol(t,x,param),tspan,x0,ODEoptions);   % true model
[tS,xS]=ode15s(@(t,x)sparseGalerkin(t,x,Xi_S,polyorder,usesine),tspan,x0,ODEoptions);  % SINDY approximate
[tB,xB]=ode15s(@(t,x)sparseGalerkin(t,x,Xi_B,polyorder,usesine),tspan,x0,ODEoptions);  % BINDY approximate
[tR,xR]=ode15s(@(t,x)sparseGalerkin(t,x,Xi_R,polyorder,usesine),tspan,x0,ODEoptions);  % BINDY approximate

dxA=zeros(size(xA));
for i=1:length(tA)
    dxA(i,:)=vanderpol(tA(i),xA(i,:),param);
end
dxS=zeros(size(xS));
for i=1:length(tS)
    dxS(i,:)=sparseGalerkin(tS(i),xS(i,:)',Xi_S,polyorder,usesine);
end
dxB=zeros(size(xB));
for i=1:length(tB)
    dxB(i,:)=sparseGalerkin(tB(i),xB(i,:)',Xi_B,polyorder,usesine);
end
dxR=zeros(size(xR));
for i=1:length(tR)
    dxR(i,:)=sparseGalerkin(tR(i),xR(i,:)',Xi_R,polyorder,usesine);
end

%% Generating Figure 3
f7=figure('Position',[100   370   400   350]);hold on;
clr = colororder('gem');
plot(xA(:,1),xA(:,2),'k-','LineWidth',1.5);
plot(x(:,1),x(:,2),'kx','MarkerSize',12);
plot(xS(:,1),xS(:,2),'-','LineWidth',1,'Color',clr(1,:));
plot(xB(:,1),xB(:,2),'-','LineWidth',2,'Color',clr(2,:));
plot(xR(:,1),xR(:,2),'-','LineWidth',1,'Color',clr(3,:));
hold off;
% title('Finite Difference');
axis([-2.5 2.5 -10 10]);
a=gca;box on;
set(a,'FontSize',16);
xlabel('$${x}_1$$','Interpreter','latex','FontSize',20);
ylabel('$${x}_2$$','Interpreter','latex','FontSize',20);
legend('Truth','Noisy Data','SINDy','B-SINDy','SparseBayes','FontSize',14,'location','southeast');
if savefig
    saveas(f7,[figpath 'VanDerPol_Large_x1x2_FD_sigma' num2str(sigma) '_dt' num2str(dt) '.fig']);
    saveas(f7,[figpath 'VanDerPol_Large_x1x2_FD_sigma' num2str(sigma) '_dt' num2str(dt) '.svg']);
end

f8=figure('Position',[100   370   400   350]);hold on;
clr = colororder('gem');
plot(dxA(:,1),dxA(:,2),'k-','LineWidth',1.5);
plot(D1*x(:,1),D1*x(:,2),'kx','MarkerSize',12);
plot(dxS(:,1),dxS(:,2),'-','LineWidth',1,'Color',clr(1,:));
plot(dxB(:,1),dxB(:,2),'-','LineWidth',2,'Color',clr(2,:));
plot(dxR(:,1),dxR(:,2),'-','LineWidth',1,'Color',clr(3,:));
hold off;
% title('Finite Difference');
axis([-8 8 -40 40]);
a=gca;box on;
set(a,'FontSize',16);
xlabel('$$\dot{x}_1$$','Interpreter','latex','FontSize',20);
ylabel('$$\dot{x}_2$$','Interpreter','latex','FontSize',20);
% legend('Truth','Noisy Data','SINDy','B-SINDy','SparseBayes','FontSize',14,'location','southeast');
if savefig
    saveas(f8,[figpath 'VanDerPol_Large_dx1dx2_FD_sigma' num2str(sigma) '_dt' num2str(dt) '.fig']);
    saveas(f8,[figpath 'VanDerPol_Large_dx1dx2_FD_sigma' num2str(sigma) '_dt' num2str(dt) '.svg']);
end