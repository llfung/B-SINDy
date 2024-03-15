function dy = lotkavolterra(t,y,param)

dy = [
param.alpha*y(1)+param.beta*y(1)*y(2);
param.gamma*y(2)+param.delta*y(1)*y(2);
];

