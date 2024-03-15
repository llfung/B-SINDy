function Xi=wkn2Xi(inds,w_kn,theta_size)
    %% Reformating parameters (w_kn) into Sparse Matrix (Xi) given indices of parameters (inds)
    % INPUT ARGUMENT
    % 
    % inds   Index of non-zero parameters in the library
    %          Can be of sparsity pattern (logical) or 
    %          indices form (numbers)
    %
    % w_kn   Parameter values in dense form
    %
    % theta_size   Number of terms in the library
    %
    % OUTPUT ARGUMENT
    %
    % Xi     Sparse Matrix of parameters
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Initialisation
    if nargin>2 % theta_size as an optional input
        Xi_height=theta_size;
    else
        Xi_height=max(inds);
    end
    % Initialise Sparse Matrix
    Xi=zeros(Xi_height,size(w_kn,2));
    
    % Switch between sparsity pattern or indices 
    if islogical(inds)
        % Sparsity pattern to indices
        inds_=1:Xi_height;
        inds_=inds_(inds);
    elseif isnumeric(inds)
        % Copy the indices
        inds_=inds;
    else
        error('Unknown index type in inds');
    end

    % Indices to Sparse Matrix
    for i=1:length(inds_)
        Xi(inds_(i),:)=w_kn(i,:);
    end
end
