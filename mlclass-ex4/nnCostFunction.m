function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



X = [ones(m,1) X]; %Appending bias units
a1 = X;
a2 = sigmoid(a1 * Theta1');
a2 = [ones(m,1) a2];
h_theta = sigmoid(a2 * Theta2');%dimensions = m * num_of_labels

yk = zeros(m, num_labels);
for i=1:m,
    yk(i, y(i)) = 1; % y is a m*1 matrix, for each i y(i) represents the label, i.e., y(1) = 4 means for the set X(4,:) the output label is 4 , so in yk, i'll make the column4 of row1 as 1 and every other column of that row a zero.
end
   
J = J + sum(sum(-yk.*log(h_theta)-(1-yk).*log(1-h_theta)));
J = J/m;

%----------------------------------------------------------------------------------------------
%Regularization

%We should not be regularizing the terms that correspond to the bias. 
%For the matrices Theta1 and Theta2, this corresponds to the ﬁrst column of each matrix.

reg = sum(sum(Theta1.^2)) - sum(Theta1.^2)(1,1); %Square element wise and sum on that gives a row vector, take sum of that and remove unrequired first column sum

reg = reg + sum(sum(Theta2.^2)) - sum(Theta2.^2)(1,1); % same for Theta2

J = J + (lambda/(2*m))*reg;


%-----------------------------------------------------------------------------------------------
%Backprop

for u=1:m,
    %forward prop
    a1 = X(u,:);% X already has bias units added
    a1 = a1';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2]; %adding bias unit
    z3 = Theta2 * a2;
    h_theta = sigmoid(z3);
    
    %calculating gradients of nodes
    delta3 = h_theta - (yk(u,:))'; %uth row gives ground truth values of uth example
    delta2 = (Theta2' * delta3) .*sigmoidGradient([1; z2]);%1 is added to z2 inorder to match dimensions as left op of .* has one row more, as it has bias units corresponding value too
    delta2 = delta2(2:end); %removing bias units gradient
    
    %Calculating gradients of weights/thetas
    Theta2_grad = Theta2_grad + delta3 * a2';
    Theta1_grad = Theta1_grad + delta2 * a1'; % removing the bias unit 4 lines above helped here not to get dimension error

end

Theta1_grad = Theta1_grad ./m ;
Theta2_grad = Theta2_grad ./m;

%---------------------------------------------------------------------------------------------------------------------------
%Regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end) .* (lambda/m); %first column shouldn't be regularized
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end) .* (lambda/m); %first column shouldn't be regularized















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
