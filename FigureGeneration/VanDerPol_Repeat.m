%% Success Rate of recovering the Van Der Pol System from data
%  Using Bayesian-SINDy, the original SINDy (STLS) and SparseBayes (RVM)
%  
% Copyright 2023, All Rights Reserved
% Code by Lloyd Fung and Matthew Juniper
% Based on code by Steven L. Brunton
%   For Paper, "Discovering Governing Equations from Data:
%            Sparse Identification of Nonlinear Dynamical Systems"
%   by S. L. Brunton, J. L. Proctor, and J. N. Kutz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialisation
% Initialise paralelisation
% par=parpool(); 

% clear all, close all, clc
figpath = './figs/';
addpath(genpath('./'));

% Set highest polynomial order of the combinations of polynomials of the state vector
polyorder = 3;
% Disable sin and cos of variables in the library (legacy)
usesine = 0;
% Set the parameters of the Van Der Pol System
param.beta  = 4;
% Set the number of variables in the system
D = 2; % Lorenz has 3 dimensions

%% Loop Param
% Time span of data
t_final=12;

% sigma_x: Noise level in the data
eps_x = 0.2; %(Change according to paper)

% Number of iterations to obtain the success rate
TestNum=100;
%TestNum=10000; (Used in paper)

% Threshold of Model Coefficient error to deem recovery successful
Success_MCE=0.25;

% Number of Data points (loop)
N_array=21:20:601;
%N_array=21:2:601; (Used in paper)

% Ground truth
Xi_truth = zeros(10,D);
Xi_truth(3,1)=1;
Xi_truth(2,2)=-1;
Xi_truth(3,2)=4;
Xi_truth(8,2)=-4;
% Sparsity Pattern
corr=logical(Xi_truth);

% Initialising result storage
SINDy_array = zeros(size(N_array));
BINDy_array = zeros(size(N_array));
RVM_array   = zeros(size(N_array));

%% Loop through different number of data
parfor ii=1:length(N_array)
    %% generate Data
    x0=[2 0];  % Initial condition

    % Number of data points 
    N=N_array(ii);

    % Time 
    dt=t_final/(N-1);
    tspan=0:dt:t_final; % BINDy is best used in low data (can be lower than # of candidate func)
    
    % Run ODE solver to generate time series data
    ODEoptions = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,D),'InitialStep',1e-6);
    [~,x_clean]=ode89(@(t,x) vanderpol(t,x,param),tspan,x0);


    %% Repeat learning
    SINDy=0;
    BINDy=0;
    RVM=0;  

    for jj=1:TestNum
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
        % Weak Formulation
        % int_pt=8;
        % P=2;
        % Q=2;
        % [I,D1]=weak(N,int_pt,P,Q,dt,5);
        
        int_pt=8;
        [I,D1]=FD(N,int_pt,dt);
        
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
        lambda = 0.4; % (Change according to eps_x)
        
        % SINDy !!
        Xi_S = sparsifyDynamics(Theta,dx,lambda,D);
    
        if all(logical(Xi_S)==corr,'all') && ((norm(Xi_S-Xi_truth)/norm(Xi_truth))<Success_MCE)
            SINDy=SINDy+1;
        end

        %% Sparse regression: Bayesian-SINDy (with Noise Iteration)
        % Assuming the Priors have zero mean and variance of
        PparamV=10^2;% Arbitrary large variance with zero mean for all coefficients
        priorA=speye(size(Theta,2))/PparamV; % Inverse of covariance in the prior of param.

        warning('off','MATLAB:nearlySingularMatrix');
        
        % Bayesian-SINDy !!
        Xi_B=BayesianRegressGreedy_NoiseIter(Theta,dx,priorA,var_dx,Theta_Var);

        if all(logical(Xi_B)==corr,'all') && ((norm(Xi_B-Xi_truth)/norm(Xi_truth))<Success_MCE)
            BINDy=BINDy+1;
        end
    
        warning('on','MATLAB:nearlySingularMatrix');
        
        %% Sparse Bayes (RVM) (Tippings 2001, 2003)
        OPTIONS		= SB2_UserOptions('iterations',10000,...
							          'diagnosticLevel', 0,...
							          'monitor', 10,...
                                      'FixedNoise',false);
    
        SETTINGS	= SB2_ParameterSettings();

        % Initialise output
        Xi_RVM = zeros(M,D);
    
        for i=1:D
            % Now run the main SPARSEBAYES function
            [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = ...
                SparseBayes('Gaussian', Theta, dx(:,i), OPTIONS, SETTINGS);
                Xi_RVM(PARAMETER.Relevant,i)	= PARAMETER.Value;
        end

        if all(logical(Xi_RVM)==corr,'all') && ((norm(Xi_RVM-Xi_truth)/norm(Xi_truth))<Success_MCE)
            RVM=RVM+1;
        end    
    end
    %% Storing results    
    SINDy_array(ii)=SINDy;
    BINDy_array(ii)=BINDy;
    RVM_array(ii)=RVM;
    disp([num2str(ii) '/' num2str(length(N_array))]);
end
%% Saving result into MAT (for server run)
% clear par;
% save(['VanDerPol_SuccessRate_eps' num2str(eps_x) '_tf' num2str(t_final) '.mat']);
% return
%% Plotting out results
savefig = false;
f=figure('Position',[100   100   400   200]);
set(gca,'FontSize',14)
hold on;
clr=colororder("gem");
plot(N_array,SINDy_array/TestNum*100,'-','LineWidth',2,'Color',clr(1,:))
plot(N_array,BINDy_array/TestNum*100,'-','LineWidth',2,'Color',clr(2,:));
plot(N_array,  RVM_array/TestNum*100,'-','LineWidth',2,'Color',clr(3,:));
xlim([0 max(N_array)]);ylim([0 100]);
ylabel('Success Rate (%)','Position',[-20.48483870967742,33.33338101704915,0]);
xlabel('Number of data points (N)');
% legend('SINDy','B-SINDy','SparseBayes','Location','southeast');

if savefig
    saveas(f,[figpath 'VanDerPol_SuccessRate_eps' num2str(eps_x) '_tf' num2str(t_final) '.fig']);
    saveas(f,[figpath 'VanDerPol_SuccessRate_eps' num2str(eps_x) '_tf' num2str(t_final) '.svg'],'svg');
end
clear f;
