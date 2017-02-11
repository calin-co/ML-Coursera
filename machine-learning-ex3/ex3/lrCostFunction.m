function [J, grad] = lrCostFunction(theta, X, y, lambda)
%Returns the cost funtction and the equations for calculating the
%partial derivates

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

hypotesis = sigmoid(X*theta);

classOne = find(y==1);
classZero = find(y==0);

cost = zeros(length(hypotesis),1);
cost(classOne) = log(hypotesis(classOne));
cost(classZero) = log ( 1 - hypotesis(classZero));
normTheta = (theta')*(theta);

% nu penalizam termenul liber
J = -1/m * sum(cost) + ( lambda/(2*m) * (normTheta - theta(1)^2) );

%vectorized implementation
diff = hypotesis - y;
unreg_grad = 1/m * (diff' * X);
grad_reg_param = lambda/m * theta;
grad_reg_param(1) = 0;
grad = unreg_grad' + grad_reg_param;

grad = grad(:);

end
