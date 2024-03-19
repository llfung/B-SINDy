function Xi = sparsifyDynamics(Theta,dXdt,lambda,n)
% Copyright 2015, All Rights Reserved
% Developed by S. L. Brunton, J. L. Proctor, and J. N. Kutz
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
%        Proceedings of the National Academy of Sciences, 113(15):3932-3937, 2016.
% Reproduced here with permission from S. L. Brunton, 2024.

% Perform Xi = pinv(Theta)*dXdt, which solves dXdt = Theta * Xi
% Xi measures the influence (on dXdt) of each column of Theta,
% averaged (regressed) over all of the timesteps. There are three columns in Xi,
% one for each of [ x , y , z ].

% compute Sparse regression: sequential least squares
Xi = Theta\dXdt;  % initial guess: Least-squares

% lambda is our sparsification knob.
for k=1:10
    smallinds = (abs(Xi)<lambda);   % find small coefficients
    Xi(smallinds)=0;                % and threshold
    for ind = 1:n                   % n is state dimension
        biginds = ~smallinds(:,ind);
        % Regress dynamics onto remaining terms to find sparse Xi
        Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind); 
    end
end
