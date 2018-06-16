function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J1 = 0; J2 = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m,1),X];
Y = zeros(m,num_labels);
for i=1:m
   if y(i)==0
       Y(i,10)=1;
   else
       Y(i,y(i))=1;
   end
end
Y = Y';
z2 = Theta1 * X';
a2 = sigmoid(z2);
a2 = [ones(1,m);a2];
z3 = Theta2 * a2;
hx = sigmoid(z3);
for i=1:m
    for k=1:num_labels
       J1 = J1 - Y(k,i)*log(hx(k,i))-(1-Y(k,i))*log(1-hx(k,i));
    end
end
J1 = J1/m;

for i=1:hidden_layer_size
    for k=1:input_layer_size
        J2 = J2 + Theta1(i,k+1)^2;
    end
end
for i=1:num_labels
    for k=1:hidden_layer_size
        J2 = J2 + Theta2(i,k+1)^2;
    end
end
J2 = J2 * lambda / 2 /m;
J = J1 + J2;

delta3 = hx - Y;
delta2 = Theta2' * delta3 ;
delta2 = delta2(2:hidden_layer_size+1,:);
delta2 = delta2 .* sigmoidGradient(z2);
dta1 = zeros(hidden_layer_size,input_layer_size+1);
dta2 = zeros(num_labels,hidden_layer_size+1);
dta1 = dta1 + delta2 * X;
dta2 = dta2 + delta3 * a2';
t1 = lambda*[zeros(hidden_layer_size,1),Theta1(:,2:input_layer_size+1)];
t2 = lambda*[zeros(num_labels,1),Theta2(:,2:hidden_layer_size+1)];
Theta1_grad = dta1/m+t1/m;
Theta2_grad = dta2/m+t2/m;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
