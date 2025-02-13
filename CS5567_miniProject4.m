% This Matlab code demonstrates how to perform image classification using transfer learning
% with VGG19 deep learning network.

% Unzip the contents of 'archive.zip' (from Kaggle) into a folder named 'archive'
unzip('archive.zip', 'archive');

% Create an ImageDatastore object 'imds' that reads image data from the 'archive' folder
% 'IncludeSubfolders' is set to true to include images in subfolders
% 'LabelSource' is set to 'foldernames' to use folder names as labels for images
% 'ReadFcn' is set to use the customReadFcn defined at the end of the script
imds = imageDatastore('archive', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames', ...
    'ReadFcn', @customReadFcn); 

% Split the image datastore into 70% training and 30% validation datastores
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% Calculate the number of training images
numTrainImages = numel(imdsTrain.Labels);

% Display a random sample of 25 training images in a 5x5 grid
idx = randperm(numTrainImages,25);
figure
for i = 1:25 
    subplot(5,5,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

% Load the pre-trained VGG19 network
net = vgg19;

% Analyze the network structure
analyzeNetwork(net)

% Get the input size of the network
inputSize = net.Layers(1).InputSize

% Remove the last three layers of the network to prepare for transfer learning
layersTransfer = net.Layers(1:end-3);

% Calculate the number of unique classes in the training data
numClasses = numel(categories(imdsTrain.Labels))

% Define the new layers for transfer learning
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Define the parameters for image augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

% Create augmented image datastores for training and validation
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network using transfer learning
netTransfer = trainNetwork(augimdsTrain,layers,options);

% Extract features from the fc7 layer
Extract features from the fc7 layer
fc7Layer = 'fc7'; % for VGG19

featuresTrain = activations(netTransfer, augimdsTrain, fc7Layer, 'OutputAs', 'columns');
featuresValidation = activations(netTransfer, augimdsValidation, fc7Layer, 'OutputAs', 'columns');

% Calculate cosine similarity between enrollment and verification images
cosineSimilarity = @(x, y) dot(x, y) / (norm(x) * norm(y));
enrollmentIdx = 1:3:size(featuresValidation, 2);
verificationIdx = [2:3:size(featuresValidation, 2), 3:3:size(featuresValidation, 2)];
scores = zeros( numel(enrollmentIdx) , numel(verificationIdx));

for i = 1:numel(enrollmentIdx)
    enrollmentFeature = featuresValidation(:, enrollmentIdx(i));
    for j = 1:size(verificationIdx, 2)
        verificationFeature = featuresValidation(:, verificationIdx(j));
        similarity = cosineSimilarity(enrollmentFeature, verificationFeature);
        scores(i,j) = similarity;
    end
end


% Create genuine and impostor score sets
genuineScores = diag(scores);
impostorScores = scores(:);
impostorScores = setdiff(impostorScores, genuineScores);

% Plot histograms of genuine and impostor scores
figure;
histogram(genuineScores, 'Normalization', 'pdf', 'BinWidth', 0.1, 'FaceColor', 'b');
hold on;
histogram(impostorScores, 'Normalization', 'pdf', 'BinWidth', 0.1, 'FaceColor', 'r');
xlabel('Cosine Similarity');
ylabel('Probability Density');
legend('Genuine Scores', 'Impostor Scores');
grid on;

% Compute and plot ROC curve
[FPR, TPR, T, AUC] = perfcurve([ones(size(genuineScores)); zeros(size(impostorScores))], ...
                               [genuineScores; impostorScores], 1);

figure;
plot(FPR, TPR);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);
grid on;

%%%%%%%% bonus %%%%%%%%

% Find the threshold for ~1% FAR
desired_FAR = 0.01;
[~, idx] = min(abs(FPR - desired_FAR));
threshold = T(idx);
GAR = TPR(idx);

fprintf('Threshold: %.4f\n', threshold);
fprintf('Training GAR: %.2f%%\n', GAR * 100);

% Apply the decision threshold to the validation scores
valid_genuine = genuineScores >= threshold;
valid_impostor = impostorScores >= threshold;

FAR_valid = sum(valid_impostor) / length(impostorScores);
GAR_valid = sum(valid_genuine) / length(genuineScores);

fprintf('Validation FAR: %.2f%%\n', FAR_valid * 100);
fprintf('Validation GAR: %.2f%%\n', GAR_valid * 100);

% Plot the ROC curve with the threshold marked and shaded area up to 0.01 FAR
figure;
plot(FPR, TPR, 'b', 'LineWidth', 2);
hold on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);

% Mark the 0.01 threshold
plot(FPR(idx), TPR(idx), 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
text(FPR(idx) + 0.01, TPR(idx) - 0.02, sprintf('Threshold: %.4f', threshold));

legend('ROC Curve', '0.01 Threshold');
grid on;

%%%%%%%%%
function data = customReadFcn(filename)
    data = imread(filename);
    data = cat(3, data, data, data);
end