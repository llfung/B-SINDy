%% Figure 10: Recovering the Lotka-Volterra equations from Lynx-Hare Pelts Trade Data
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
[Xi_B,~,m_out,Var_out]=BayesianRegressGreedy_NoiseIter(Theta,dx,priorA,var_dx,Theta_Var);

% Finding Co-variance of the learnt parameters
for i=1:D
    SIGMA{i}=inv(Theta(:,m_out{i})'*(Theta(:,m_out{i})./Var_out(:,i))+priorA(m_out{i},m_out{i}));
end

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

%% COMPARE evaluation of the learnt model
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,D));

% LearnFunc=@(x,Xi)([Xi(2,1)*x(1)+Xi(5,1)*x(1)*x(2);...
%     Xi(3,2)*x(2)+Xi(5,2)*x(1)*x(2)]);
LearnFunc=@(x,Xi)(([1 x(1) x(2) x(1)^2 x(1)*x(2) x(2)^2 x(1)^3 x(1)^2*x(2) x(1)*x(2)^2 x(2)^3]*Xi)');

[tA,xA]=ode89(@(t,x)LearnFunc(x,Xi_B),[0 20],x(1,:),options); % Approximate from BINDy
[tS,xS]=ode89(@(t,x)LearnFunc(x,Xi_S),[0 20],x(1,:),options); % Approxiimate from SINDy
[tR,xR]=ode89(@(t,x)LearnFunc(x,Xi_R),[0 20],x(1,:),options); % Approxiimate from RVM

%% Sampling from the distribution of the parameters (given by SIGMA)
TestNum=100;
tTest=0:0.05:20;
x_col=zeros(length(tTest),D,TestNum);
for i=1:TestNum
    Xi_B_=zeros(size(Xi_B));
    Xi_B_(m_out{1},1)=mvnrnd(Xi_B(m_out{1},1)',SIGMA{1})';
    Xi_B_(m_out{2},2)=mvnrnd(Xi_B(m_out{2},2)',SIGMA{2})';
    [~,x_]=ode45(@(t,x)LearnFunc(x,Xi_B_),tTest,x(1,:),options); % Approximate from BINDy
    x_col(:,:,i)=x_;
end

% Extracing 5th and 95th percentile
x5=prctile(x_col,5,3);
x95=prctile(x_col,95,3);

%% Generaing Figure 10a - Prediction from each learning result (and bound)
savefig = false;
FigObj=figure('Position',[100 100 600 400]);clr=colororder("gem");

a1=axes('Position',[0.086 0.11 0.87 0.4]);hold on;
plot([0:20]'+1900,x(:,2),'kx','MarkerSize',12,'LineWidth',1);
plot(tS+1900,xS(:,2),'-','LineWidth',1.5,'Color',clr(1,:));
plot(tA+1900,xA(:,2),'-','LineWidth',2,'Color',clr(2,:));
plot(tR+1900,xR(:,2),'-','LineWidth',1.5,'Color',clr(3,:));
patch([tTest'+1900;flipud(tTest')+1900],[x95(:,2);flipud(x5(:,2))],'r','FaceColor',clr(2,:),'FaceAlpha',0.35,'EdgeColor',clr(2,:),'LineStyle','none');
hold off;
set(gca,'FontSize',16);
ylabel('x_2: Lynx','FontSize',18,'Interpreter','tex','Position',[1899.193042262139,44.30861686610575,0]);
ylim([0 90]);
grid on;box on;
xlabel('Time (year)','FontSize',18,'Interpreter','tex','Position',[1910,-12]);
xticks([1900:2:1920]);

a2=axes('Position',[0.086 0.55 0.87 0.4]);hold on;
plot([0:20]'+1900,x(:,1),'kx','MarkerSize',12,'LineWidth',1);
plot(tS+1900,xS(:,1),'-' ,'LineWidth',1.5,'Color',clr(1,:));
plot(tA+1900,xA(:,1),'-' ,'LineWidth',2,'Color',clr(2,:));
plot(tR+1900,xR(:,1),'-' ,'LineWidth',1.5,'Color',clr(3,:));
patch([tTest'+1900;flipud(tTest')+1900],[x95(:,1);flipud(x5(:,1))],'r','FaceColor',clr(2,:),'FaceAlpha',0.35,'EdgeColor',clr(2,:),'LineStyle','none');
hold off;
set(gca,'FontSize',16);
ylabel('x_1: Hare ','FontSize',18,'Interpreter','tex','Position',[1899.307984790875,49.37111686610576]);
ylim([0 120]);yticks([0:20:120]);
grid on;box on;
legend('Data','SINDy','B-SINDy','SparseBayes','B-SINDy 95% int.','Orientation','horizontal',...
    'Position',[0.103333333333334 0.933665313789385 0.888333333333333 0.05375]);
xticks([1900:2:1920]);
xticklabels({});

if savefig
    saveas(FigObj,[figpath 'LynxHare_time.fig']);
    saveas(FigObj,[figpath 'LynxHare_time.svg'],'svg');
end
%% Generaing Figure 10b - Phase Diagram
FigObj=figure('Position',[100 100 400 400]);
a=gca; hold on;
plot(x(:,1),x(:,2),'kx','MarkerSize',12,'LineWidth',2);
plot(xS(:,1),xS(:,2),'-','LineWidth',1,'Color',clr(1,:));
plot(xA(:,1),xA(:,2),'-','LineWidth',2,'Color',clr(2,:));
plot(xR(:,1),xR(:,2),'-','LineWidth',1,'Color',clr(3,:));
hold off;
a.FontSize=16;
legend('Data',...
    'SINDy',...
    'B-SINDy',...
    'SparseBayes','FontSize',14)
xlabel('x_1: Hare ','FontSize',18,'Interpreter','tex');
ylabel('x_2: Lynx','FontSize',18,'Interpreter','tex');
ylim([0 90]);xlim([0 120]);
grid on;box on;
if savefig
    saveas(FigObj,[figpath 'LynxHare_phase.fig']);
    saveas(FigObj,[figpath 'LynxHare_phase.svg'],'svg');
end