function [I,D1,D2]=FD(N_pt,N_dom,dt)
    %% Generate Differential Matrices using Finite Differences, assuming equispacing / constant sampling frequency
    % INPUT ARGUMENT
    %
    % N_pt     Number of data points
    %
    % N_dom    Order of accuracy (Number of points used -1 / Number of domain
    %          used to obtain each derivative point)
    %
    % dt       Delta t between two adjacent points
    
    %% Initialisation (check if N_dom is supported)
    if  N_dom <=1
        error('N_dom must be positive even integer');
    elseif mod(N_dom,2)~=0
        error('Odd N_dom not yet implemented.');
    else
        %% Generate differential matrices based on the order of accuracy
        switch N_dom
            case 2
                I=spdiags(ones(N_pt,1)*[0 1 0],0:2,N_pt-2,N_pt);
                D1=spdiags(ones(N_pt,1)*[-1/2 0 1/2]/dt,0:2,N_pt-2,N_pt);
                D2=spdiags(ones(N_pt,1)*[1 -2 1]/dt^2,0:2,N_pt-2,N_pt);
            case 4
                I=spdiags(ones(N_pt,1)*[0 0 1 0 0],0:4,N_pt-4,N_pt);
                D1=spdiags(ones(N_pt,1)*[1/12 -2/3 0 2/3 -1/12]/dt,0:4,N_pt-4,N_pt);
                D2=spdiags(ones(N_pt,1)*[-1/12 4/3 -5/2 4/3 -1/12]/dt^2,0:4,N_pt-4,N_pt);
            case 6
                I=spdiags(ones(N_pt,1)*[0 0 0 1 0 0 0],0:6,N_pt-6,N_pt);
                D1=spdiags(ones(N_pt,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60]/dt,0:6,N_pt-6,N_pt);
                D2=spdiags(ones(N_pt,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90]/dt^2,0:6,N_pt-6,N_pt);
            case 8
                I=spdiags(ones(N_pt,1)*[0 0 0 0 1 0 0 0 0],0:8,N_pt-8,N_pt);
                D1=spdiags(ones(N_pt,1)*[1/280 -4/105 1/5 -4/5 0 4/5 -1/5 4/105 -1/280]/dt,0:8,N_pt-8,N_pt);
                D2=spdiags(ones(N_pt,1)*[-1/560 8/315 -1/5 8/5 -205/72 8/5 -1/5 8/315 -1/560]/dt^2,0:8,N_pt-8,N_pt);
            case 10
                I=spdiags(ones(N_pt,1)*[0 0 0 0 0 1 0 0 0 0 0],0:10,N_pt-10,N_pt);
                D1=spdiags(ones(N_pt,1)*[-1/1260 5/504 -5/84 5/21 -5/6 0 5/6 -5/21 5/84 -5/504 1/1260]/dt,0:10,N_pt-10,N_pt);
                D2=spdiags(ones(N_pt,1)*[1/3150 -5/1008 5/126 -5/21 5/3 -5269/1800 5/3 -5/21 5/126 -5/1008 1/3150]/dt^2,0:10,N_pt-10,N_pt);
            otherwise
                Coeff0 = zeros(1,N_dom+1);Coeff0(1+N_dom/2)=1;
                [CoeffD1,CoeffD2] = CentralFD(N_dom);
                I=spdiags(ones(N_pt,1)*Coeff0,0:N_dom,N_pt-N_dom,N_pt);
                D1=spdiags(ones(N_pt,1)*CoeffD1/dt,0:N_dom,N_pt-N_dom,N_pt);
                D2=spdiags(ones(N_pt,1)*CoeffD2/dt^2,0:N_dom,N_pt-N_dom,N_pt);
        end
    end
end

%% Internal utility functions
function [CoeffD1,CoeffD2] = CentralFD(n)
    %% 1st and 2nd order Central finite difference Coefficients
    %   Inputs:  
    %       n : order of accuracy (even)
    %
    %   Outputs:
    %       coefs: central finite difference coefficients.
    %
    %   NB: Stable for n<20

    MAT=power(-n/2:n/2,(0:n)'); 
    CoeffD1=MAT\[0;1;zeros(n-1,1)];
    CoeffD2=MAT\[0;0;2;zeros(n-2,1)];
    % Rounding near zeros to zero
    CoeffD1(abs(CoeffD1)<10000*eps)=0;
    CoeffD2(abs(CoeffD1)<10000*eps)=0;
    % Transposing
    CoeffD1=CoeffD1';
    CoeffD2=CoeffD2';
end
