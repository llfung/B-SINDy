function dy = lorenz(t,y,param,b,r)
    if nargin>3
        s=param;
    else
        s=param.sigma;
        b=param.beta;
        r=param.rho;
    end
    
    dy = [
    s*(y(2)-y(1));
    y(1)*(r-y(3))-y(2);
    y(1)*y(2)-b*y(3);
    ];
end