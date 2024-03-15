function var = VarPNormal(mu,sigma2,order)
%% Variance of x^order when x ~ N(mu,sigma2)
    var = ENPNormal(mu,sigma2,order*2)-(ENPNormal(mu,sigma2,order)).^2;
end
function E = ENPNormal(mu,sigma2,order)
%% Expectation of x^order when x ~ N(mu,sigma2)
switch order
    case 1
        E=mu;
    case 2
        E=mu.^2+sigma2;
    case 3
        E=mu.^3+3*mu.*sigma2;
    case 4
        E=mu.^4+6*mu.^2.*sigma2+3*sigma2.^2;
    case 5
        E=mu.^5+10*mu.^3.*sigma2+15*mu.*sigma2.^2;
    case 6
        E=mu.^6+15*mu.^4.*sigma2+45*mu.^2.*sigma2.^2+15*sigma2.^3;
    case 7
        E=mu.^7+21*mu.^5.*sigma2+105*mu.^3.*sigma2.^2+105*mu.*sigma2.^3;
    case 8
        E=mu.^8+28*mu.^6.*sigma2+210*mu.^4.*sigma2.^2+420*mu.^2.*sigma2.^3+105*sigma2.^4;
    otherwise
        error('Power of random variable too high! Not supported by this code.')
end
end