
inputSize = 28*28;



hiddenSizeL1 = 225;    % Layer 1 Hidden Size
hiddenSizeL2 = 100;    % Layer 2 Hidden Size
hiddenSizeL3 = 64;
hiddenSizeL4 = 16;
hiddenSizeL5 = 1;

sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter
beta = 3;              % weight of sparsity penalty term

epsilon = 0.1;
%%======================================================================
%% STEP 1: Load data from the MNIST database

trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');

sigma = trainData * trainData' / size(trainData,2);
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
trainData = ZCAWhite * trainData;
%%======================================================================
%% STEP 2a: Train the first sparse autoencoder

fprintf('Training first sparse autoencoder.\n');
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

addpath minFunc/
options.Method  = 'ldfgs';
options.MaxIter = 100;
options.display = 'on';

[sae1OptTheta, cost] = minFunc(@(t)sparseAutoencoderCost(t, ...
                               inputSize, hiddenSizeL1, ...
                               lambda, sparsityParam, beta, trainData),...
                               sae1Theta, options); 
                           
% -------------------------------------------------------------------------
figure(1);
W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
display_network(W1');


%%======================================================================
%% STEP 2b: Train the second sparse autoencoder
fprintf('Training second sparse autoencoder.\n');
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

[sae2OptTheta, cost] = minFunc(@(t)sparseAutoencoderCost(t, ...
                               hiddenSizeL1, hiddenSizeL2, ...
                               lambda, sparsityParam, beta, sae1Features),...
                               sae2Theta, options); 

% -------------------------------------------------------------------------
figure(2);
W2 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
display_network(W2');

%%======================================================================
%% STEP 2c: Train the third sparse autoencoder
fprintf('Training third sparse autoencoder.\n');
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
sae3Theta = initializeParameters(hiddenSizeL3, hiddenSizeL2);

[sae3OptTheta, cost] = minFunc(@(t)sparseAutoencoderCost(t, ...
                               hiddenSizeL2, hiddenSizeL3, ...
                               lambda, sparsityParam, beta, sae2Features),...
                               sae3Theta, options); 

% -------------------------------------------------------------------------
figure(3);
W3 = reshape(sae3OptTheta(1:hiddenSizeL3 * hiddenSizeL2), hiddenSizeL3, hiddenSizeL2);
display_network(W3');
% 
% %%======================================================================
% %% STEP 2d: Train the forth sparse autoencoder
% 
% [sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                         inputSize, trainData);
% sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
% 
% [sae2OptTheta, cost] = minFunc(@(t)sparseAutoencoderCost(t, ...
%                                hiddenSizeL1, hiddenSizeL2, ...
%                                lambda, sparsityParam, beta, sae1Features),...
%                                sae2Theta, options); 
% 
% % -------------------------------------------------------------------------
% W2 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
% display_network(W2');
% 
% 
% %%======================================================================
% %% STEP 2e: Train the fifth sparse autoencoder
% 
% [sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                         inputSize, trainData);
% sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
% 
% [sae2OptTheta, cost] = minFunc(@(t)sparseAutoencoderCost(t, ...
%                                hiddenSizeL1, hiddenSizeL2, ...
%                                lambda, sparsityParam, beta, sae1Features),...
%                                sae2Theta, options); 
% 
% % -------------------------------------------------------------------------
% W2 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
% display_network(W2');
%%======================================================================
%% STEP 3: Train the softmax classifier
% 
% [sae3Features] = feedForwardAutoencoder(sae3OptTheta, hiddenSizeL3, ...
%                                         hiddenSizeL2, sae2Features);
% saeSoftmaxTheta = 0.005 * randn(hiddenSizeL3 * numClasses, 1);
% 
% tempLambda = 1e-4;
% softmaxModel = softmaxTrain(hiddenSizeL3, numClasses, ...
%                             tempLambda, sae3Features, ...
%                             trainLabels, options);
% 
% saeSoftmaxOptTheta = softmaxModel.optTheta(:);
% 
% %%======================================================================
% %% STEP 5: Finetune softmax model

stack = cell(3,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
stack{3}.w = reshape(sae3OptTheta(1:hiddenSizeL3*hiddenSizeL2), ...
                     hiddenSizeL3, hiddenSizeL2);
stack{3}.b = sae3OptTheta(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);

exW = reshape(sae3OptTheta(1 + hiddenSizeL3*hiddenSizeL2 : 2 * hiddenSizeL3*hiddenSizeL2), ...
                     hiddenSizeL2, hiddenSizeL3);
exb = sae3OptTheta(2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3 + 1:end);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta =  stackparams ;

[stackedAEOptTheta, cost] = minFunc(@(t)myStackedAECost(t, ...
                               inputSize, netconfig, ...
                               lambda, sparsityParam, beta, ...
                               trainData),...
                               stackedAETheta, options); 

% %%======================================================================
% %% STEP 6: Test 
% 
% testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
% testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
% 
% testLabels(testLabels == 0) = 10; % Remap 0 to 10
% 
% [pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
%                           numClasses, netconfig, testData);
% 
% acc = mean(testLabels(:) == pred(:));
% fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
% 
% [pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
%                           numClasses, netconfig, testData);
% 
% acc = mean(testLabels(:) == pred(:));
% fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
% 
% % Accuracy is the proportion of correctly classified images
% % The results for our implementation were:
% %
% % Before Finetuning Test Accuracy: 87.7%
% % After Finetuning Test Accuracy:  97.6%
