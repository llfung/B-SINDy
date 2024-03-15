function [Xi,J_Evi_out,m_out,Var_out,m2nd]=BayesianRegressGreedy_NoiseIter(Theta,dx,priorA,Var_dx,Var_Theta,options)
    %% Bayesian-SINDy - (Gaussian) Bayesian Inference and Model Selection by type-II maximum likelihood (evidence maximisation)
    %  with noise iteration process to estimate noise variance 
    %  given known variance in the function library of the regressor (Var_Theta)
    %  and response (dx, aka target)
    % 
    % INPUT ARGUMENTS:
    % 
    %	Theta	Library of Candidate functions
    %
    %	dx		Target: time derivative of the variables
    %
    %	priorA		Inverse Prior variance for each variable
    % 
    %	Var_dx		Noise variance of time derivatives
    % 
    %	Var_Theta	Noise variance of each column in the library 
    %
    %   options     Options for fine tuning noise iteration 
    %
    % OUTPUT ARGUMENTS:
    %
    %   Xi     Output of estimated parameter (MAP point)
    %
    %   J_Evi_out  Negative Log-Evidence of the selected model
    %
    %   m_out  Index of the non-zero terms in the selected model
    % 
    %   Var_out    Total noise variance estimate based on Xi
    %
    %   m2nd   For Analysis: Index of the non-zero terms in the 2nd best model

    % Copyright 2024, Code by Lloyd Fung
    % For Paper, "Rapid Bayesian identification of sparse nonlinear dynamics from scarce and noisy data"
    % by L. Fung, U. Fasel and M. P. Juniper
    %
    % This file is part of the B-SINDy package.
    % See "LICENSE" and "README.md" in the package for details about
    % license and copyright of the B-SINDy package.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Setting defaults for options
    if nargin <= 5
        options.MaxNoiseIter = 10; % Maximum noise iteration 
        options.NoiseIterThres = 1e-4; % Threshold for relative change in w for stopping noise iteration 
    end
    if ~isfield(options,'MaxNoiseIter')
        options.MaxNoiseIter = 10; % Maximum noise iteration 
    end
    if ~isfield(options,'NoiseIterThres')
        options.NoiseIterThres = 1e-4; % Threshold for relative change in w for stopping noise iteration 
    end

    %% Initialisation
    % Extract the number of data points
    N = size(Theta,1);
    % Extract the number of terms in the Theta library
    M = size(Theta,2);
    % Extract the number of state variables
    D = size(dx,2);

    % Initialising arrays for storage
    Xi=zeros(M,D);
    J_Evi_out=zeros(1,D);
    m_out=cell(1,D);
    Var_out=zeros(N,D);
    
    % Initialise figure size 
    if options.LogEvidenceAlphaPlot
        fig_obj=figure('Position',[100 100 800 350]);
    end

    %% Loop through each dimension
    for d=1:D
        m=1:M; % initialise indices of terms in the model (include all terms)
        % Initialise temporary array for storage
            m_j=cell(M,1);
            w_j=cell(M,1);
          Var_j=cell(M,1);
        J_Evi_j=zeros(1,M);
        % Greedy search: iterate by removing terms one-by-one in the model
        for j=1:M
            J_Evi_i=zeros(1,(M-j+1));
            w_i=zeros((M-j),(M-j+1));
            Var_i = zeros(N,(M-j+1));
            % Loop through all terms one can remove from current model
            for i=1:(M-j+1)
                % Remove one term (the ith term from the current list of
                % indices m)
                m_i=m([1:i-1,i+1:end]);
                % Select the Theta column according to indices selected (m_i)
                Theta_=Theta(:,m_i);
                % Select the prior variance according to indices selected (m_i)
                A_=priorA(m_i,m_i);
                % Number of terms currently in the model
                M_=length(m_i);
                % Find parameter value for the MAP point - Same as linear regression! (Prior serve as regularisation)
                w = (Theta_'*(Theta_./Var_dx(:,d))+A_)\(Theta_'*(dx(:,d)./Var_dx(:,d)));
                % Iterate w to get the more accurate estimate for overall
                % noise variance sigma 
                for ii = 1:options.MaxNoiseIter
                    w_prev=w;
                    % sigma^2 = sigma^2_(dt) + sigma^2_(Theta)*w
                    Var = Var_dx(:,d) + Var_Theta(:,m_i)*(w.^2);
                    % Reformulate SIGMAInv
                    SIGMAInv=Theta_'*(Theta_./Var)+A_;
                    % Update MAP estimate for w
                    w = SIGMAInv\(Theta_'*(dx(:,d)./Var));
                    % Stop iteration when change in w is below a threshold
                    if norm(w_prev-w)/norm(w_prev) < options.NoiseIterThres
                        break;
                    end
                end
                % Negative Log-Posterior at MAP point
                J_MAP = sum((dx(:,d)-Theta_*w).^2./Var,1)/2+log(2*pi)*N/2+sum(log(Var)/2) ...
                    + (w')*A_*w/2+log(2*pi)*M_/2-log(det(A_))/2;
                
                % Negative Log-Evidence, by Laplace Approximation
                J_Evi_i(i)=J_MAP+log(det(SIGMAInv/2/pi))/2;
                
                % Storing parameter estimate and total noise variance estimate
                  w_i(:,i)=w;
                Var_i(:,i)=Var;
            end
            % Select the index with the MINIMUM J (maximum posterior likelihood).
            % This means: "when you drop the polynomial term with this index, the remaining model
            % has higher evidence than all the other alternative models."
            [~,i_min]=min(J_Evi_i);

            % Remove this index from the indices that can be selected next
            m=m([1:i_min-1,i_min+1:end]);

            % Storing values to array
                m_j{j}=m; % History of terms selected
                w_j{j}=w_i(:,i_min); % History of parameters of terms selected
            J_Evi_j(j)=J_Evi_i(i_min); % History of resulting neg. log-evidence 
              Var_j{j}=Var_i(:,i_min); % History of total noise variance estimate
        end
        % Selects the MODEL with the minimum J, which is the maximum evidence 
        [~,j_min]=min(J_Evi_j);
        % Format the parameter estimate back into a sparse matrix
        Xi(:,d)=wkn2Xi(m_j{j_min},w_j{j_min},M);
        % Outputting the resulting neg. log-evidence, index of selected
        % terms and total noise variance
        J_Evi_out(d)=J_Evi_j(j_min);
            m_out{d}=m_j{j_min};
        Var_out(:,d)=Var_j{j_min};

        % For Analysis: Outputting the index of the term selected in the
        % 2nd best model
        if j_min>1
            m2nd{d} = m_j{j_min-1};
        else
            m2nd{d} = m_j{j_min};
        end
    end
end