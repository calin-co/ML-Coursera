function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%The cost function J is defined as
% J:= 1/m * sum ( y * (-log(h(x) ) + (1 - y) * (1 - log(h(x)) ) 
classOne = find(y==1);
classZero = find(y==0);

hypotesis = sigmoid(X*theta) ;

cost = zeros(length(hypotesis),1);
cost(classOne) = log(hypotesis(classOne));
cost(classZero) = log ( 1 - hypotesis(classZero));

J = -1/m * sum(cost) ;
for i = 1:length(grad)
    grad(i) = 1/m * ( hypotesis - y)' * X(:,i) ; 
% =============================================================

end
