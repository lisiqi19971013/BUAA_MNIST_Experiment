clear;clc;
data_train = loadMNISTImages('train-images.idx3-ubyte');
labels_train = loadMNISTLabels('train-labels.idx1-ubyte');
data_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
data_train = data_train(:,1:20000);
labels_train = labels_train(1:20000);
data_test = data_test(:,1:300);
labels_test = labels_test(1:300);

N = 784;  
K = 100;% can be any other value  
n = length(data_train);  
[p,m] = size(data_test);  
testResults = zeros(1,m);
compLabel = zeros(1,K);  
 
for i=1:m  
    img = repmat(data_test(:,i),1,n);
    for j = 1:n
        sum = 0;
        for k=1:N
            sum = sum + (img(k,j)-data_train(k,j))^2;
        end
        comp(j) = sqrt(sum);
    end  
    [sortedComp,index] = sort(comp);  
    for j = 1:K  
        compLabel(j) = labels_train(index(j));  
    end  
    table = tabulate(compLabel);  
    [maxCount,num] = max(table(:,2));  
    testResults(i) = table(num);
    fprintf('No.%d Test image   ||   ', i);
    fprintf('Test Result: %d   ||   Real Category: %d   ||   ', testResults(i),labels_test(i));
    if (testResults(i) == labels_test(i))
        fprintf('Correct!\n');
    else
        fprintf('Wrong!\n');
    end
end  

error=0;  
for i=1:m  
  if (testResults(i) ~= labels_test(i))  
    error=error+1;  
  end  
end  
error = error/m*100;  
fprintf('\nTraining Set Accuracy: %f\n', 100-error);