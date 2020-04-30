function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   g = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end
