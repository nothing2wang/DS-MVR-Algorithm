function [ MSE ] = MSE_measure( X,Xt )
% X: estimated data
% Xt: ground truth
% the purpose is to match the columns and measure the error

X = X*diag(1./sqrt(sum(X.^2)));

Xt = Xt*diag(1./(sqrt(sum(Xt.^2))+eps));

M = X'*Xt;

MSE_col = max(M);
MSE = abs(mean(2- 2*MSE_col));




end

