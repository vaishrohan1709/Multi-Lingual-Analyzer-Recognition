function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%feedforward
netcost=0;
for i=1:m
	% feedforward to find hypothesis for each training example
	a1 = X(i,:)';
	a1 = [1;a1];    %adding bias unit to activation vector of layer1
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1;a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);   % a3 = hyp(x)
	yVec = zeros(num_labels,1);
	yVec(y(i)) = 1;
	cost = yVec' * log(a3) + (1 .- yVec)' * log(1 .- a3);  % cost for training example i
	netcost = netcost + cost;
end;
J = -1/m * netcost;

regularization = (lambda/(2*m)) * (sum(sum((Theta1(:,2:end)).^2)) + sum(sum((Theta2(:,2:end)).^2)));

J = J + regularization;

%backpropagation algorithm

%delta1 = zeros(hidden_layer_size,input_layer_size+1);
%delta2 = zeros(num_labels,hidden_layer_size+1);

for i=1:m
	a1 = X(i,:)';
	a1 = [1;a1];    %adding bias unit to activation vector of layer1
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1;a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);   % a3 = hyp(x)
	yVec = zeros(num_labels,1);
	yVec(y(i)) = 1;
	err3 = a3 - yVec; %error corresponding to output layer
	err2 = (Theta2' * err3) .* (a2 .* (1 - a2));
	% err1 does not exist because there can be no errors in input layer
	
	err2 = err2(2:end);  % ignore or remove sigma2(0)
	
	Theta2_grad = Theta2_grad + err3 * a2';
	Theta1_grad = Theta1_grad + err2 * a1';
	
end;
Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;	

%bias term remains as it is
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
