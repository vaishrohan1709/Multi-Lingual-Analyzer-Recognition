function W = debugInitializeWeights(fan_out, fan_in)

%   W should be set to a matrix of size(1 + fan_in, fan_out) as
%   the first row of W handles the "bias" terms

% Set W to zeros
W = zeros(fan_out, 1 + fan_in);

% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;

end
