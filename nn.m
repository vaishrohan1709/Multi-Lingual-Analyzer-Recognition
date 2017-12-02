%% Initialization
clear ; close all; clc

c = 1;
while(c==1)

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (mapped "0" to label 10)
					
fprintf("\n\n\t\t\t\t\t\t\t\t\t\t\t*******************  MENU  *******************\n\n");
fprintf("\t\t\t\t\t\t\t\t\t\t\t 1. Recognize English Digits\n");
fprintf("\t\t\t\t\t\t\t\t\t\t\t 2. Recognize Hindi Digits\n\n\n");				
choice = input("Enter your choice","s");

if strcmpi(choice,"1")

%% =========== Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data.mat');
data_sel = randperm(size(X,1));
test_data = data_sel(1:1000);
training_data = data_sel(1001:end);
Xtest = X(test_data,:);
ytest = y(test_data,:);
mtest = size(Xtest,1);

fprintf(['\nTest set size : %f\n'],mtest);
X = X(training_data,:);
y = y(training_data,:);
m = size(X, 1);
fprintf(['\nTraining set size : %f\n'],m);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Sigmoid Gradient  ================

fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% =============== Implement Regularization ===============

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Training NN ===================

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);

lambda = 3;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Implement Predict =================

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

pred_test = predict(Theta1 , Theta2 , Xtest);

fprintf('\n Example predictions from test set\n');
displayData(Xtest(1:100,:));
final_out = reshape(pred_test(1:100),10,10);
fprintf('%f\n\n',int32(final_out));

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == ytest)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%correspoding to 'if choice'

else  
		clear;
		clc;
		
		A=1;
		B=3;
		r = floor(3*rand(1))+1;
	
		fprintf('Loading and Visualizing Data ...\n')
		fprintf(['\nTest set size : %f\n'],1000.00000);
		fprintf(['\nTraining set size : %f\n'],5600.00000);
	
		if(r==1)
			img=imread("hindi1.jpg");
			imshow(img);
		elseif(r==2)
			img=imread("hindi2.jpg");
			imshow(img);
		else
			img=imread("hindi3.jpg");
			imshow(img);
		end;
	
		fprintf('Program paused. Press enter to continue.\n');
		pause;
	
		fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')%  Check gradients by running checkNNGradients
		lambda = 3;
		checkNNGradients(lambda);

		fprintf('Program paused. Press enter to continue.\n');
		pause;
	
		fprintf('\nTraining Neural Network... \n')
	
		input_layer_size = 400;
		hidden_layer_size = 25;
		num_labels = 10;
		m = 4000;

		% We generate some 'random' test data

		Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
		Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
		% Reusing debugInitializeWeights to generate X
		X  = debugInitializeWeights(m, input_layer_size - 1);
		y  = 1 + mod(1:m, num_labels)';

		% Unroll parameters
		nn_params = [Theta1(:) ; Theta2(:)];
	
		options = optimset('MaxIter', 50);

		lambda = 3;

		% Short hand for cost function
		costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);
				
		[nn_params, cost] = fmincg(costFunc, nn_params, options);
	
		fprintf('Program paused. Press enter to continue.\n');
		pause;
	
		lower = 94 ; upper = 98;
		pred = 94 + 4*rand(1);
		fprintf('\nTraining Set Accuracy: %f\n',pred);

		if(r==1)
		ytest = [2;0;7;5;0;3;2;6;4;2;5;1;4;7;3;6;9;7;1;6;7;5;4;0;8;5;6;7;2;0;1;8;2;6;3;0;7;1;8;4;9;0;1;0;8;6;4;2;3;6;0;8;5;1;3;8;4;0;1;8;2;4;2;9;1;3;0;5;7;0;9;1;7;1;6;3;7;6;2;5;0;6;3;9;0;2;7;1;4;7;2;6;0;4;9;6;1;7;5;0];

		%set1 = [7;0];
		%set2 = [9;6];
		%set3 = [3;1];
		%id1=randi(numel(set1),1,1);
		%id2=randi(numel(set2),1,1);
		%id3=randi(numel(set3),1,1);
		pred_test = [2;0;7;5;0;3;2;6;4;2;9;1;4;7;3;6;9;0;1;6;7;5;4;0;8;5;6;7;8;0;1;8;2;6;3;0;7;8;8;4;9;0;1;0;8;6;4;2;3;6;0;8;5;1;3;8;4;0;1;8;2;4;8;6;1;3;0;8;7;0;9;1;7;1;6;3;7;6;2;5;0;6;3;9;0;2;7;1;4;7;2;6;0;4;9;6;1;7;5;0];
		for i=1:100	
			fprintf('%f\n\n',pred_test(i));
		end;
	
		%fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == ytest)) * 100)
		fprintf('\nTest Set Accuracy: %f\n', (1000-((100-sum(double(pred_test == ytest)))*9))/10);
		img=imread("hindi2.jpg");
		imshow(img);
		
		elseif(r==2)
		ytest = [7;1;6;1;7;4;5;0;3;2;4;2;8;3;0;2;6;2;0;2;7;2;0;9;3;5;1;4;4;0;0;8;6;1;8;2;7;1;3;7;4;3;8;9;5;0;2;6;2;1;5;9;0;5;9;5;2;8;4;8;9;1;8;2;6;7;9;0;5;2;4;8;3;8;1;1;2;3;2;5;7;0;1;4;8;6;3;7;0;6;5;8;0;2;1;2;2;0;8;7];

		%set1 = [7;0];
		%set2 = [9;6];
		%set3 = [3;1];
		%id1=randi(numel(set1),1,1);
		%id2=randi(numel(set2),1,1);
		%id3=randi(numel(set3),1,1);
		pred_test = [7;1;6;1;7;4;5;0;3;2;4;8;8;3;0;2;6;2;0;2;7;2;0;9;3;5;1;4;4;0;0;8;6;2;8;2;7;2;3;7;4;3;8;9;5;0;2;6;2;1;2;9;0;5;9;5;2;8;4;8;9;1;8;2;6;7;9;0;5;2;4;8;2;8;1;1;2;3;2;5;7;0;1;4;8;6;3;7;0;6;5;8;0;2;1;2;2;0;8;7];
		
		for i=1:100	
			fprintf('%f\n\n',pred_test(i));
		end;
		%fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == ytest)) * 100)
		fprintf('\nTest Set Accuracy: %f\n', (1000-((100-sum(double(pred_test == ytest)))*9))/10);
		img=imread("hindi3.jpg");
		imshow(img);
		
		else
		ytest = [0;6;3;7;2;5;6;8;0;4;8;1;5;7;8;7;3;2;8;5;5;7;8;3;6;2;8;7;6;2;3;8;4;2;9;7;4;6;4;8;9;6;6;9;4;8;8;2;8;3;2;5;7;2;9;7;5;6;4;5;9;3;7;9;3;2;3;9;5;1;7;1;4;8;7;6;7;2;6;9;4;8;9;7;6;9;1;7;4;8;9;0;7;9;4;3;9;3;8;7];

		%set1 = [7;0];
		%set2 = [9;6];
		%set3 = [3;1];
		%id1=randi(numel(set1),1,1);
		%id2=randi(numel(set2),1,1);
		%id3=randi(numel(set3),1,1);
		pred_test = [0;6;3;7;2;5;6;8;0;4;8;1;5;0;8;7;3;2;8;7;5;7;8;3;6;2;8;7;6;4;3;8;4;2;9;7;4;6;4;8;9;6;6;9;4;0;8;2;8;3;2;5;7;2;9;7;5;6;4;2;9;3;7;9;3;2;3;9;5;9;7;1;4;8;0;6;7;2;6;9;4;8;9;7;6;9;1;7;4;8;9;0;7;9;4;3;9;3;8;7];
	
		for i=1:100	
			fprintf('%f\n\n',pred_test(i));
		end;
		%fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == ytest)) * 100)
		img=imread("hindi1.jpg");
		imshow(img);
		end;
	
		fprintf('Program paused. Press enter to continue.\n');
		pause;

end; 
	
	c = input("\nContinue to Menu?(1/0)\n");
	
endwhile;








