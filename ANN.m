clear;clc;
data_train = loadMNISTImages('train-images.idx3-ubyte');
labels_train = loadMNISTLabels('train-labels.idx1-ubyte');
data_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
data_train = data_train(:,1:6000);
labels_train = labels_train(1:6000);
data_test = data_test(:,1:1000);
labels_test = labels_test(1:1000);

input_layer_size  = 784;  
hidden_layer_size = 100; 
num_labels = 10;   

X=data_train';
m = size(X, 1);
y = labels_train;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
lambda = 1;
ep = 200;
options = optimset('MaxIter', ep);
costFunction = @(p) nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels, X, y, lambda);
[nn_params, cost, Acc] = fmincg(costFunction, initial_nn_params, options);
figure(1);
plot(1:ep,cost);
title('cost-epochs');xlabel('epochs');ylabel('cost');
figure(2);
plot(1:ep,100 - Acc);
title('Error-epochs');xlabel('epochs');ylabel('Error/%');
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
pred = predict(Theta1, Theta2, data_test');
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == labels_test)) * 100);