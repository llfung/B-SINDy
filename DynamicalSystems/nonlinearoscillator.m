function dy = nonlinearoscillator(t,y,param)

dy = [
param.alpha*y(1)^3+param.beta*y(2)^3;
param.gamma*y(1)^3+param.delta*y(2)^3
];

