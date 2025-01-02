#include "../include/ForwardNeuralNetwork.h"


// Constructor
ForwardNeuralNetwork::ForwardNeuralNetwork(int inputSize_,
                                           int hiddenSize_,
                                           int outputSize_,
                                           double learningRate_,
                                           unique_ptr<ActivationFunction> hiddenActivation,
                                           unique_ptr<ActivationFunction> outputActivation)
    : inputSize(inputSize_), hiddenSize(hiddenSize_), outputSize(outputSize_),
      learningRate(learningRate_), dis(-1.0, 1.0) // Initialize weights between -1 and 1
{
    // Initialize random number generator with random device
    random_device rd;
    gen = mt19937(rd());

    // Assign activation functions
    activationHiddenFunc = std::move(hiddenActivation);
    activationOutputFunc = std::move(outputActivation);

    // Initialize weights and biases
    initializeWeights();
}

// Initialize Weights and Biases
void ForwardNeuralNetwork::initializeWeights() {
    // Initialize weights from input to hidden layer
    weightsInputHidden.resize(hiddenSize, vector<double>(inputSize));
    for(auto &row : weightsInputHidden)
        for(auto &w : row)
            w = dis(gen);

    // Initialize biases for hidden layer
    biasHidden.resize(hiddenSize);
    for(auto &b : biasHidden)
        b = dis(gen);

    // Initialize weights from hidden to output layer
    weightsHiddenOutput.resize(outputSize, vector<double>(hiddenSize));
    for(auto &row : weightsHiddenOutput)
        for(auto &w : row)
            w = dis(gen);

    // Initialize biases for output layer
    biasOutput.resize(outputSize);
    for(auto &b : biasOutput)
        b = dis(gen);

        // Initialize gradients with zeros
           dWeightsInputHidden.resize(hiddenSize, std::vector<double>(inputSize, 0.0));
           dBiasHidden.resize(hiddenSize, 0.0);
           dWeightsHiddenOutput.resize(outputSize, std::vector<double>(hiddenSize, 0.0));
           dBiasOutput.resize(outputSize, 0.0);

           // Optional: Verify initialization
           assert((int)dWeightsInputHidden.size() == hiddenSize && "dWeightsInputHidden size mismatch.");
           for(const auto &row : dWeightsInputHidden)
               assert((int)row.size() == inputSize && "dWeightsInputHidden row size mismatch.");

           assert((int)dWeightsHiddenOutput.size() == outputSize && "dWeightsHiddenOutput size mismatch.");
           for(const auto &row : dWeightsHiddenOutput)
               assert((int)row.size() == hiddenSize && "dWeightsHiddenOutput row size mismatch.");

}

// Forward Propagation Implementation
vector<double> ForwardNeuralNetwork::forward(const vector<double> &input) {
    assert(input.size() == static_cast<size_t>(inputSize) && "Input size does not match network input size.");

    // Compute z-values and activations for hidden layer
    zHidden.resize(hiddenSize);
    activationHidden.resize(hiddenSize);
    for(int i = 0; i < hiddenSize; ++i) {
        zHidden[i] = biasHidden[i];
        for(int j = 0; j < inputSize; ++j)
            zHidden[i] += weightsInputHidden[i][j] * input[j];
        activationHidden[i] = activationHiddenFunc->activate(zHidden[i]);
    }

    // Compute z-values and activations for output layer
    zOutput.resize(outputSize);
    activationOutput.resize(outputSize);
    for(int i = 0; i < outputSize; ++i) {
        zOutput[i] = biasOutput[i];
        for(int j = 0; j < hiddenSize; ++j)
            zOutput[i] += weightsHiddenOutput[i][j] * activationHidden[j];
        activationOutput[i] = activationOutputFunc->activate(zOutput[i]);
    }

    return activationOutput;
}

// Compute Mean Squared Error
double ForwardNeuralNetwork::computeLoss(const vector<double> &predictions, const vector<double> &targets) const {
    assert(predictions.size() == targets.size() && "Size mismatch between predictions and targets.");
    double mse = 0.0;
    for(size_t i = 0; i < predictions.size(); ++i)
        mse += pow(predictions[i] - targets[i], 2);
    return mse / predictions.size();
}

// Backward Propagation Implementation
void ForwardNeuralNetwork::backward(const vector<double> &input, const vector<double> &target) {
    assert(target.size() == static_cast<size_t>(outputSize) && "Target size does not match network output size.");

    // Calculate output error (delta)
    vector<double> deltaOutput(outputSize, 0.0);
    for(int i = 0; i < outputSize; ++i) {
        // For MSE loss: derivative is (output - target) * activation_derivative(z)
        double error = activationOutput[i] - target[i];
        deltaOutput[i] = error * activationOutputFunc->derivative(zOutput[i]);
    }

    // Calculate hidden layer error (delta)
    vector<double> deltaHidden(hiddenSize, 0.0);
    for(int i = 0; i < hiddenSize; ++i) {
        double error = 0.0;
        for(int j = 0; j < outputSize; ++j)
            error += deltaOutput[j] * weightsHiddenOutput[j][i];
        deltaHidden[i] = error * activationHiddenFunc->derivative(zHidden[i]);
    }

    // Compute gradients for weightsHiddenOutput and biasOutput
    for(int i = 0; i < outputSize; ++i) {
        for(int j = 0; j < hiddenSize; ++j)
            dWeightsHiddenOutput[i][j] += deltaOutput[i] * activationHidden[j];
        dBiasOutput[i] += deltaOutput[i];
    }

    // Compute gradients for weightsInputHidden and biasHidden
    for(int i = 0; i < hiddenSize; ++i) {
        for(int j = 0; j < inputSize; ++j)
            dWeightsInputHidden[i][j] += deltaHidden[i] * input[j];
        dBiasHidden[i] += deltaHidden[i];
    }
}

// Update Parameters Implementation
void ForwardNeuralNetwork::updateParameters() {
    // Update weightsHiddenOutput and biasOutput
    for(int i = 0; i < outputSize; ++i) {
        for(int j = 0; j < hiddenSize; ++j)
            weightsHiddenOutput[i][j] -= learningRate * dWeightsHiddenOutput[i][j];
        biasOutput[i] -= learningRate * dBiasOutput[i];
    }

    // Update weightsInputHidden and biasHidden
    for(int i = 0; i < hiddenSize; ++i) {
        for(int j = 0; j < inputSize; ++j)
            weightsInputHidden[i][j] -= learningRate * dWeightsInputHidden[i][j];
        biasHidden[i] -= learningRate * dBiasHidden[i];
    }

    // Reset gradients to zero after updating
    for(int i = 0; i < outputSize; ++i) {
        for(int j = 0; j < hiddenSize; ++j)
            dWeightsHiddenOutput[i][j] = 0.0;
        dBiasOutput[i] = 0.0;
    }

    for(int i = 0; i < hiddenSize; ++i) {
        for(int j = 0; j < inputSize; ++j)
            dWeightsInputHidden[i][j] = 0.0;
        dBiasHidden[i] = 0.0;
    }
}

// Compute Mean Absolute Error
double ForwardNeuralNetwork::computeMAE(const vector<double> &predictions, const vector<double> &targets) const {
    assert(predictions.size() == targets.size() && "Size mismatch between predictions and targets.");
    double mae = 0.0;
    for(size_t i = 0; i < predictions.size(); ++i)
        mae += abs(predictions[i] - targets[i]);
    return mae / predictions.size();
}

vector<double> ForwardNeuralNetwork::predict(const vector<vector<double> > &inputs) {
    vector<double> predictions;
    for(const auto &input : inputs) {
        vector<double> prediction = forward(input);
        predictions.push_back(prediction[0]); // Assuming single output
    }
    return predictions;
}

vector<double> ForwardNeuralNetwork::flatten(const vector<vector<double> > &list) {
    vector<double> flattenedList;
    for(const auto &row : list) {
        flattenedList.push_back(row[0]); // Assuming single target
    }
    return flattenedList;
}

// Training Method Implementation with Batch Size
void ForwardNeuralNetwork::train(const vector<vector<double> > &trainInputs,
                                const vector<vector<double> > &trainTargets,
                                const vector<vector<double> > &testInputs,
                                const vector<vector<double> > &testTargets,
                                int epochs,
                                int batchSize,
                                vector<double> &totalTrainMAE,
                                vector<double> &totalTestMAE) {
    assert(trainInputs.size() == trainTargets.size() && "Number of inputs and targets must match.");

    int dataSize = trainInputs.size();

    // Initialize a vector of indices for shuffling

    vector<int> indices(dataSize);
    for(int i = 0; i < dataSize; ++i) {
        indices[i] = i;
    }

    for(int epoch = 1; epoch <= epochs; ++epoch) {
        double totalLoss = 0.0;

        // Shuffle the data at the beginning of each epoch for better convergence
        random_device rd_shuffle;
        mt19937 gen_shuffle(rd_shuffle());
        shuffle(indices.begin(), indices.end(), gen_shuffle);

        // Iterate over the dataset in batches

        for(int batchStart = 0; batchStart < dataSize; batchStart += batchSize) {


            // Determine the actual batch size (handles the last batch if it's smaller)
            int currentBatchSize = min(batchSize, dataSize - batchStart);

            // Reset gradients before processing the current batch
            for(int i = 0; i < outputSize; ++i) {
                for(int j = 0; j < hiddenSize; ++j) {
                    dWeightsHiddenOutput[i][j] = 0.0;
                dBiasOutput[i] = 0.0;
                }
            }

            for(int i = 0; i < hiddenSize; ++i) {
                for(int j = 0; j < inputSize; ++j) {
                    dWeightsInputHidden[i][j] = 0.0;
                dBiasHidden[i] = 0.0;
                }
            }

            // Process each sample in the current batch
            for(int i = 0; i < currentBatchSize; ++i) {
                int idx = batchStart + i;
                int actualIdx = indices[idx]; // Get the shuffled index

                // Forward pass
                vector<double> prediction = forward(trainInputs[actualIdx]);

                // Compute loss for this example
                double loss = computeLoss(prediction, trainTargets[actualIdx]);
                totalLoss += loss;

                // Backward pass
                backward(trainInputs[actualIdx], trainTargets[actualIdx]);
            }

            // Update parameters after processing the batch
            updateParameters();
        }

        // Average loss over the dataset
        double averageLoss = totalLoss / dataSize;

        vector<double> flattenedTestTargets = flatten(testTargets);
        vector<double> flattenTrainTargets = flatten(trainTargets);

        vector<double> testPredictions = predict(testInputs);
        vector<double> trainPredictions = predict(trainInputs);

        double trainMAE = computeMAE(trainPredictions, flattenTrainTargets);
        double testMAE = computeMAE(testPredictions, flattenedTestTargets);

        totalTrainMAE.push_back(trainMAE);
        totalTestMAE.push_back(testMAE);

        // Display progress
        if(epoch % 100 == 0 || epoch == 1) { // Adjust frequency as needed
            cout << "Epoch " << epoch << " - Loss: " << averageLoss << endl;
        }
    }
}

// Compute Mean Squared Error
double ForwardNeuralNetwork::computeMSE(const vector<double> &predictions, const vector<double> &targets) const {
    if (predictions.size() != targets.size()) {
        std::cerr << "Size mismatch between predictions and targets:\n"
                  << "  predictions.size() = " << predictions.size() << "\n"
                  << "  targets.size()     = " << targets.size() << std::endl;
        assert(false && "Size mismatch between predictions and targets.");
    }
    double mse = 0.0;
    for(size_t i = 0; i < predictions.size(); ++i)
        mse += pow(predictions[i] - targets[i], 2);
    return mse / predictions.size();
}

// Compute Root Mean Squared Error
double ForwardNeuralNetwork::computeRMSE(const vector<double> &predictions, const vector<double> &targets) const {
    return sqrt(computeMSE(predictions, targets));
}

// Compute R-squared
double ForwardNeuralNetwork::computeRSquared(const vector<double> &predictions, const vector<double> &targets) const {
    assert(predictions.size() == targets.size() && "Size mismatch between predictions and targets.");
    double meanTarget = accumulate(targets.begin(), targets.end(), 0.0) / targets.size();

    double ssTotal = 0.0;
    double ssResidual = 0.0;
    for(size_t i = 0; i < targets.size(); ++i) {
        ssTotal += pow(targets[i] - meanTarget, 2);
        ssResidual += pow(targets[i] - predictions[i], 2);
    }

    return 1 - (ssResidual / ssTotal);
}
