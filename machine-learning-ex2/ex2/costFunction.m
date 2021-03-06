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
%

h_theta_x = sigmoid(X*theta);

log_h_theta_x = log(h_theta_x);
one_minus_h_theta_x = 1-h_theta_x;
log_one_minus_h_theta_x = log(one_minus_h_theta_x);


first_term = transpose(y)*log_h_theta_x;
second_term = transpose(1-y)*log_one_minus_h_theta_x;

J = -1.0*(first_term + second_term)/m;


h_theta_x_minus_y = h_theta_x - y;

grad = (transpose(X)*h_theta_x_minus_y)./m;


% =============================================================

end
