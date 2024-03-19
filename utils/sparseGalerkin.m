function dy = sparseGalerkin(t,y,ahat,polyorder,usesine)
% Copyright 2015, All Rights Reserved
% Developed by S. L. Brunton, J. L. Proctor, and J. N. Kutz
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
%        Proceedings of the National Academy of Sciences, 113(15):3932-3937, 2016.
% Reproduced here with permission from S. L. Brunton, 2024.

yPool = poolData(y',length(y),polyorder,usesine);
dy = (yPool*ahat)';