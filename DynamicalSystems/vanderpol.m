function dy = vanderpol(t,y,param)

dy = [
y(2);
param.beta*y(2)*(1-y(1)^2)-y(1);
];

