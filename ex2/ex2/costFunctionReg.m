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

%Calculate h(x) parameterterized by theta
g = X * theta;

h = sigmoid(g);


%first sum

a=((-y)')*(log(h));
   
   %second sum
   
   b=((1-y)')*(log(1-h));

J = (a-b)/m;

%Calculate the regularized term

R = (sum( theta(2:end).^2))*((lambda)/(2*m));

J = J + R;





%Calculate the gradient

grad=((((h-y)')*(X))')./m;
	   reg_term= theta*(lambda/m);
	   reg_term(1)=0;
	   grad = grad + reg_term;
	   
% =============================================================

end
