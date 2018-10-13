function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h_theta_x = sigmoid(X*theta);

log_h_theta_x = log(h_theta_x);
one_minus_h_theta_x = 1-h_theta_x;
log_one_minus_h_theta_x = log(one_minus_h_theta_x);


first_term = transpose(y)*log_h_theta_x;
second_term = transpose(1-y)*log_one_minus_h_theta_x;

untheta = theta(2:end);

reg_term = ((lambda/(2*m))*sum(untheta.^2));

J = (1/m)*sum(-y.*log_h_theta_x - (1-y).*(log_one_minus_h_theta_x)) + reg_term;

h_theta_x_minus_y = h_theta_x - y;
grad = ((transpose(X)*h_theta_x_minus_y)./m);

grad_term_reg = (lambda/m).*(untheta);

grad(2:end) = grad(2:end) + grad_term_reg;


% =============================================================

end
