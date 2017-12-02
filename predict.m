function p = predict(Theta1, Theta2, Xtest)
%PREDICT Predict the label of an input given a trained neural network

m = size(Xtest, 1);
num_labels = size(Theta2, 1);
 
p = zeros(size(Xtest, 1), 1);

h1 = sigmoid([ones(m, 1) Xtest] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

% =========================================================================

end
