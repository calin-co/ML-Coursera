function [final_theta, J_history] = logisticGradientDescent(X, y, alpha, max_iter)
% for a logistic regression problem this function calculates
% the optimal values for theta in order to create a decision boundary
% hypotesis is interpreted as the probabily of y = 1 given x,thetaX
J_history = zeros(1,max_iter);
dim = size(X);
theta = rands(dim(2),1);
m = dim(1);
updated_theta = zeros(length(theta), 1);

for num_iter = 1:max_iter
    hypotesis = sigmoid(X*theta);
    
    for i = 1:length(updated_theta)
        updated_theta(i) = theta(i) - alpha/m * (hypotesis - y)' * X(:,i) ; 
    end
    theta = updated_theta ;
    
    J_history(num_iter) = computeCost(X,y,theta);
end
final_theta = theta;
end