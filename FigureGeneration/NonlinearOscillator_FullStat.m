%% Fig 7: Breakdown of contribution to Total Variance sigma^2
%  Using Bayesian-SINDy and the original SINDy (STLS)
%
% Copyright 2024, All Rights Reserved
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
rng(7);

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

%% generate Data
x0=[1,0];  % Initial condition
% Time 
dt=0.1;
t_final=5; 
tspan=0:dt:t_final;

% Number of data points 
N_tilde = length(tspan);

% Run ODE solver to generate time series data
ODEoptions = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,D),'InitialStep',1e-6);
[t,x_clean]=ode89(@(t,x) nonlinearoscillator(t,x,param),tspan,x0,ODEoptions);

% Ground truth
Xi_truth = zeros(10,D);
Xi_truth( 7,:) = [param.alpha param.gamma];
Xi_truth(10,:) = [param.beta param.delta];

%% Compute Derivative and Add noise
% Set size of random noise
eps_x = 0.01;
% initialize dx
dx_clean = NaN(N_tilde,D);
% Calculate the derivative dx at each point along the trajectory
for i=1:N_tilde
    dx_clean(i,:) = nonlinearoscillator(0,x_clean(i,:),param);
end
% Add random noise to x, repeatedly to generate N_Sample samples
N_Sample=1000;
x = repmat(x_clean,[1,1,N_Sample]) + eps_x*randn(N_tilde,D,N_Sample);

%% Build library of nonlinear time series
% If each row of x contains [ x , y , z ] then each row of Theta contains (for polyorder=3):
% [1 , x , y , z , xx , xy , xz , yy , yz , zz , xxx , xxy , xxz , xyy , xyz , xzz , yyy , yyz , yzz , zzz ]

% Initialise
Theta_tilde=NaN(N_tilde,10,N_Sample);
Theta_tildeVar=NaN(N_tilde,10,N_Sample);
% For each Sample
for i=1:N_Sample
    % Library of polynomials of state variables (x,y) for each sample data
    Theta_tilde(:,:,i) = poolData(x(:,:,i),D,polyorder,usesine);
    % Estimated Variance of the library of polynomials for each sample data
    Theta_tildeVar(:,:,i) = poolDataVar(x(:,:,i),eps_x^2*ones(size(x,1),size(x,2)),polyorder);
end
% Library of polynomials of state variables (x,y) for the clean data
% (for comparison)
Theta_xclean = poolData(x_clean,D,polyorder,usesine);
% Save the number of polynomial combinations
M = size(Theta_tilde,2);

%% Derivatives
int_pt=6;
P=4;
Q=2;
[I,D1]=weak(N_tilde,int_pt,P,Q,dt);

% int_pt=6;
% [I,D1]=FD(N_tilde,int_pt,dt);

% dx from noisy sample data (x N_Sample)
dx=pagemtimes(full(D1),x);
% Variance of noise in the time derivative
var_dx=(D1).^2*eps_x^2*ones(size(x,1),size(x,2));

% Theta from noisy sample data (x N_Sample)
Theta=pagemtimes(full(I),Theta_tilde);
% Theta Variance estimation from noisy sample data (x N_Sample)
Theta_Var=pagemtimes(full(I.^2),Theta_tildeVar);

% New number of data-points
N=N_tilde-int_pt;

%% Computing real variance based on sampling (for comparison)
% Using the truth coefficients
var_real=var(dx-pagemtimes(Theta,Xi_truth),0,3);

%% Defining time array for plotting
ind=floor((int_pt+1)/2);
tPlot=t(ind+1:end-ind);

%% Sparse regression: Bayesian-SINDy (with Noise Iteration)
% Assuming the Priors have zero mean and variance of
PparamV=1^2;% Arbitrary large variance with zero mean for all coefficients
priorA=speye(size(Theta,2))/PparamV; % Inverse of covariance in the prior of param.

% Initialise
var_learn=NaN(size(dx,1),D,N_Sample);
Xi_learn = NaN(M,D,N_Sample);

corr=zeros(10,2);corr(7,:)=1;corr(10,:)=1;corr=logical(corr);
corr_learn=false(N_Sample,1);
for i=1:N_Sample
    % Bayesian-SINDy !! (Perform regression)
    [Xi,~,~,dxV_out]=BayesianRegressGreedy_NoiseIter(Theta(:,:,i),dx(:,:,i),priorA,var_dx,Theta_Var(:,:,i));
    % Parameter estimated by B-SINDy
    Xi_learn(:,:,i)=Xi;
    % Total Variance (sigma^2) based on the parameter estimated
    var_learn(:,:,i)=dxV_out;
    
    % Detect if the estimation has the correct sparsity
    if logical(Xi)==corr
        corr_learn(i)=true;
    end
end

% Extracting variance of relevant terms
var7_learn=reshape(Theta_Var(:,7,:),[],N_Sample);
var10_learn=reshape(Theta_Var(:,10,:),[],N_Sample);
%% Generate Figure 7a: Variance breakdown and comparison for the dynamics of x_1
savefig = false;
FigObj=figure(1);
FigObj.Position=[100 100 600 300];
a=gca;
hold on;

% Plot the estimated Total Variance (sigma^2), supposedly equal the sum of the next three plots
plot(tPlot,reshape(var_learn(:,1,corr_learn),[],sum(corr_learn)),'LineStyle',':','Color', [.7 .7 .7]);
% Plot the real Total Variance of the sampled data (for comparison)
p1=plot(tPlot,var_real(:,1),'k-','LineWidth',3.0);

% Plot the estimated variance from column 7 (x_1^3) of library Theta
plot(tPlot,var7_learn(:,corr_learn).*reshape(Xi_learn(7,1,corr_learn).^2,1,[]),'LineStyle',':','Color', [0.7 0.7 1]);
% Plot the real variance from column 7 (x_1^3) of the sampled data (for comparison)
p2=plot(tPlot,var(Theta(:,7,:)*param.alpha,[],3),'Color', [0 0 1],'LineWidth',3.0);


% Plot the estimated variance from column 10 (x_2^3) of library Theta
plot(tPlot,var10_learn(:,corr_learn).*reshape(Xi_learn(10,1,corr_learn).^2,1,[]),'LineStyle',':','Color', [1 .7 .7]);
% Plot the real variance from column 10 (x_2^3) of the sampled data (for comparison)
p3=plot(tPlot,var(Theta(:,10,:)*param.beta,[],3),'Color', [1 0 0],'LineWidth',3.0);

% Plot the estimated variance for dx
plot(tPlot,var_dx(:,1),'LineStyle',':','Color', [.7 1 .7]);
% Plot the real variance for dx from the sampled data (for comparison)
p4=plot(tPlot,var(dx(:,1,:),[],3),'Color', [0 1 0],'LineWidth',3.0);

% Make figure pretty and labelling
hold off;
a.FontSize=18;
xlabel('$$t$$','Interpreter','latex','FontSize',22);
ylabel('Variance $$\sigma^2$$','Interpreter','latex','FontSize',22);
legend([p1,p2,p3,p4],...
    {'$$\sigma^2$$','$$L_I \sigma^2_{x_1^3} w^2_{x_1^3}$$',...
    '$$L_I \sigma^2_{x_2^3} w^2_{x_2^3}$$','$$\sigma^2_{\partial t}$$'},...
    'interpreter','latex','FontSize',18)
if savefig
    saveas(FigObj,[figpath 'NonlinearOsc_x1_var_t_eps' num2str(eps_x) '_dt' num2str(dt) '.fig']);
    saveas(FigObj,[figpath 'NonlinearOsc_x1_var_t_eps' num2str(eps_x) '_dt' num2str(dt) '.svg'],'svg');
end

%% Generate Figure 7b: Variance breakdown and comparison for the dynamics of x_2
% (Comments same as above, not repeated)
FigObj=figure(2);
FigObj.Position=[100 100 600 300];
a=gca;
hold on;
plot(tPlot,reshape(var_learn(:,2,corr_learn),[],sum(corr_learn)),'LineStyle',':','Color', [.7 .7 .7]);
p1=plot(tPlot,var_real(:,2),'k-','LineWidth',3.0);

plot(tPlot,var7_learn(:,corr_learn).*reshape(Xi_learn(7,2,corr_learn).^2,1,[]),'LineStyle',':','Color', [0.7 0.7 1]);
p2=plot(tPlot,var(Theta(:,7,:)*param.gamma,[],3),'Color', [0 0 1],'LineWidth',3.0);

plot(tPlot,var10_learn(:,corr_learn).*reshape(Xi_learn(10,2,corr_learn).^2,1,[]),'LineStyle',':','Color', [1 .7 .7]);
p3=plot(tPlot,var(Theta(:,10,:)*param.delta,[],3),'Color', [1 0 0],'LineWidth',3.0);

plot(tPlot,var_dx(:,2),'LineStyle',':','Color', [.7 1 .7]);
p4=plot(tPlot,var(dx(:,2,:),[],3),'Color', [0 1 0],'LineWidth',3.0);
hold off;
a.FontSize=18;
xlabel('$$t$$','Interpreter','latex','FontSize',22);
ylabel('Variance $$\sigma^2$$','Interpreter','latex','FontSize',22);
legend([p1,p2,p3,p4],...
    {'$$\sigma^2$$','$$L_I \sigma^2_{x_1^3} w^2_{x_1^3}$$',...
    '$$L_I \sigma^2_{x_2^3} w^2_{x_2^3}$$','$$\sigma^2_{\partial t}$$'},...
    'interpreter','latex','FontSize',18)
if savefig
    saveas(FigObj,[figpath 'NonlinearOsc_x2_var_t_eps' num2str(eps_x) '_dt' num2str(dt) '.fig']);
    saveas(FigObj,[figpath 'NonlinearOsc_x2_var_t_eps' num2str(eps_x) '_dt' num2str(dt) '.svg'],'svg');
end