%% Fig 2: Weak Formulation VS Finite Difference in Van Der Pol
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

% Run ODE solver to generate time series data
ODEoptions = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,D),'InitialStep',1e-6);
[~,x_clean]=ode15s(@(t,x) vanderpol(t,x,param),tspan,x0,ODEoptions);


N_tilde=length(tspan);
dx_clean=NaN(size(x_clean));

for i=1:N_tilde
    dx_clean(i,:)=vanderpol(tspan(i),x_clean(i,:),param);
end

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
int_pt=8;

% Weak Formulation 
[I_weak,D_weak]=weak(N_tilde,int_pt,2,2,dt,5);
% Finite Differences
[I_FD  ,D_FD  ]=FD(N_tilde,int_pt,dt);

t = tspan(int_pt/2+1:end-int_pt/2);

Theta_tilde = poolData(x_clean,D,polyorder,usesine);
Theta_weak = I_weak*Theta_tilde;
Theta_FD = I_FD*Theta_tilde;

%% Plots Prep.: Weak VS FD
[tplot,xplot]=ode15s(@(t,x) vanderpol(t,x,param),[0 t_final],x0,ODEoptions);
dxplot=xplot;
for i=1:length(tplot)
    dxplot(i,:)=vanderpol(tplot(i),xplot(i,:),param);
end

%% Plots: Weak VS FD
savefig = false;
f1=figure('Position',[100 100 190 150]);
plot(t,D_weak*x(:,1),'x',tplot,xplot(:,2),'k-');
% legend(['Weak (noise \sigma=' num2str(eps_x) ')'],'Truth','Location','southeast');
axis([0 t_final -8 8]);
set(gca,'FontSize',14);
xlabel('$$t$$','Interpreter','latex','FontSize',16);
ylabel('$$\dot{x}_1$$','Interpreter','latex','FontSize',16);
% title(['Weak Formulation; \Deltat=' num2str(dt)],'FontSize',12);
set(gca,'Position',[0.1842    0.2067    0.7897    0.6738]);
a=gca;
a.YLabel.Position=[-1.3717    0.1702   -1.0000];
a.XLabel.Position=[6.5000  -10.4340   -1.0000];
if savefig
    saveas(f1,[figpath 'VanDerPol_x1t_Weak_sigma' num2str(sigma) '_dt' num2str(dt) '.fig']);
    saveas(f1,[figpath 'VanDerPol_x1t_Weak_sigma' num2str(sigma) '_dt' num2str(dt) '.svg']);
end

f2=figure('Position',[100 100 190 150]);
plot(t,D_FD*x(:,1),'x',tplot,xplot(:,2),'k-');
% legend(['F.D. (noise \sigma=' num2str(eps_x) ')'],'Truth','Location','southeast');
axis([0 t_final -8 8]);
set(gca,'FontSize',14);
xlabel('$$t$$','Interpreter','latex','FontSize',16);
ylabel('$$\dot{x}_1$$','Interpreter','latex','FontSize',16);
% title(['Finite Difference; \Deltat=' num2str(dt)],'FontSize',12);
set(gca,'Position',[0.1842    0.2067    0.7897    0.6738]);
a=gca;
a.YLabel.Position=[-1.3717    0.1702   -1.0000];
a.XLabel.Position=[6.5000  -10.4340   -1.0000];
if savefig
    saveas(f2,[figpath 'VanDerPol_x1t_FD_sigma' num2str(sigma) '_dt' num2str(dt) '.fig']);
    saveas(f2,[figpath 'VanDerPol_x1t_FD_sigma' num2str(sigma) '_dt' num2str(dt) '.svg']);
end

f3=figure('Position',[100   370   190   160]);hold on;
plot(D_weak*x(:,1),D_weak*x(:,2),'x:');
plot(dxplot(:,1),dxplot(:,2),'k-')
hold off;
% title('Weak Formulation');
axis([-8 8 -40 40]);
a=gca;box on;
set(a,'FontSize',14);
xlabel('$$\dot{x}_1$$','Interpreter','latex','FontSize',16);
ylabel('$$\dot{x}_2$$','Interpreter','latex','FontSize',16);
a.Position=[0.2000    0.2687    0.7739    0.6687];
% a.XLabel.Position=[-0.7417  -53.0513   -1.0000];
a.YLabel.Position=[-9.8177    1.4954   -1.0000];
if savefig
    saveas(f3,[figpath 'VanDerPol_x1x2_Weak_sigma' num2str(sigma) '_dt' num2str(dt) '.fig']);
    saveas(f3,[figpath 'VanDerPol_x1x2_Weak_sigma' num2str(sigma) '_dt' num2str(dt) '.svg']);
end

f4=figure('Position',[100   370   190   160]);hold on;
plot(D_FD*x(:,1),D_FD*x(:,2),'x:');
plot(dxplot(:,1),dxplot(:,2),'k-')
hold off;
% title('Finite Difference');
axis([-8 8 -40 40]);
a=gca;box on;
set(a,'FontSize',14);
xlabel('$$\dot{x}_1$$','Interpreter','latex','FontSize',16);
ylabel('$$\dot{x}_2$$','Interpreter','latex','FontSize',16);
a.Position=[0.2000    0.2687    0.7739    0.6687];
% a.XLabel.Position=[-0.7417  -53.0513   -1.0000];
a.YLabel.Position=[-9.8177    1.4954   -1.0000];
if savefig
    saveas(f4,[figpath 'VanDerPol_x1x2_FD_sigma' num2str(sigma) '_dt' num2str(dt) '.fig']);
    saveas(f4,[figpath 'VanDerPol_x1x2_FD_sigma' num2str(sigma) '_dt' num2str(dt) '.svg']);
end

