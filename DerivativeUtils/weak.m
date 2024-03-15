function [I,D1,D2]=weak(N_pt,N_dom,P,Q,dt,NCMaxOrder)
    %% Generate Differential Matrices using Finite Differences, assuming equispacing / constant sampling frequency
    % INPUT ARGUMENT
    %
    % N_pt     Number of data points
    %
    % N_dom    Order of accuracy (Number of points used -1 / Number of domain
    %          used to obtain each derivative point)
    %
    % P, Q     Test Function phi(t)=(t^Q-1)^P for t in [-1,1]
    %
    % dt       Delta t between two adjacent points
    %
    % NCMaxOrder   Maximum Order for integration using Newton-Cote Method
    %              (automatically selected to be the highest possible
    %              given N_dom, unless specified by NCMaxOrder)
    %              Should be less than 5.

    if nargin<6
        NCMaxOrder=2; % Higher is better for high P or Q, but should be less than 5 (Higher order may be unstable).
    end
    if mod(Q,2)~=0
        error('Q must be an even integer');
    end
    
    % t in [-1,1]
    t=(-1:2/(N_dom):1)'; 
    % Test Function phi(t)
    testfun=((t.^Q-1).^P);
    % 1st Derivative of Test Function phi'(t) 
    % (times rescaling due to dt and N_dom)
    testfun_dt=(-P*(t.^Q-1).^(P-1).*t.^(Q-1)*Q)*(2/(dt*N_dom));
    % 2nd Derivative of Test Function phi'(t) 
    % (times rescaling due to dt and N_dom)
    testfun_ddt=(P*Q*t.^(Q-2).*((t.^Q-1).^(P-2)).*(1-t.^Q+Q*(P*t.^Q-1)))*(2/(dt*N_dom))^2;

    % Find the appropiate Newton-Cotes order.
    % If the Max order is not appropiate for N_dom,
    % decrease it till one appropiate one is found.
    for NCOrder=NCMaxOrder:-1:1
        if mod(N_dom,NCOrder)==0
            break;
        end
    end

    % Create weights for integrating over [-1,1]
    SubInt=N_dom/NCOrder;
    w_in=NewtonCotes(NCOrder)';
    w_start=w_in(1);
    w_end=w_in(end);
    w_rep=w_in(2:end);w_rep(end)=w_start+w_end;
    w = [w_start; repmat(w_rep,SubInt,1)];w(end)=w_end;
    w = w*2/SubInt;


    % Represent the weak formulation as linear operators (matrices)
    I=spdiags(...
        ones(N_pt,1)*...
        (w.*testfun)',...
        0:N_dom,N_pt-N_dom,N_pt);

    D1=spdiags(...
        ones(N_pt,1)*...
        (w.*testfun_dt)',...
        0:N_dom,N_pt-N_dom,N_pt);

    D2=spdiags(...
        ones(N_pt,1)*...
        (w.*testfun_ddt)',...
        0:N_dom,N_pt-N_dom,N_pt);
end

%% Internal utility function
function w=NewtonCotes(n)
    %% Newton-Cotes coefficients for intergration with equispaced points
    % INPUT ARGUMENT
    %
    % n    Order of accuracy (Number of points used -1 / Number of domain
    %      used to obtain each derivative point)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    switch n
        case 1
            w=[1/2 1/2];
        case 2
            w=[1/6 2/3 1/6];
        case 3
            w=[1/8 3/8 3/8 1/8];
        case 4
            w=[7/90 16/45 2/15 16/45 7/90];
        case 5
            w=[19/288 25/96 25/144 25/144 25/96 19/288];
        % case 6 (may be unstable)
        %     w=[41/840 9/35 9/280 34/105 9/280 9/35 41/840];
        otherwise
            error('Integration Order for Newton-Cotes method is too high');
    end
end