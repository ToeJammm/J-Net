#pragma once

#include <vector>
#include <memory>
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include <numeric>
#include "ActivationFunction.h"

using namespace std;

// Forward Neural Network with One Hidden Layer
class ForwardNeuralNetwork {
private:
    // Network architecture
    int inputSize;
    int hiddenSize;
    int outputSize;

    // Weights and biases
    vector<vector<double> > weightsInputHidden; // [hiddenSize][inputSize]
    vector<double> biasHidden;                      // [hiddenSize]
    vector<vector<double> > weightsHiddenOutput; // [outputSize][hiddenSize]
    vector<double> biasOutput;                      // [outputSize]

    // Activations and z-values
    vector<double> activationHidden; // [hiddenSize]
    vector<double> zHidden;          // [hiddenSize]
    vector<double> activationOutput; // [outputSize]
    vector<double> zOutput;          // [outputSize]

    // Activation Functions
    unique_ptr<ActivationFunction> activationHiddenFunc;
    unique_ptr<ActivationFunction> activationOutputFunc;

    // Learning rate
    double learningRate;

    // Random number generator for weight initialization
    mt19937 gen;
    uniform_real_distribution<> dis;

    // Gradients
        vector<vector<double> > dWeightsInputHidden; // [hiddenSize][inputSize]
        vector<double> dBiasHidden;                      // [hiddenSize]
        vector<vector<double> > dWeightsHiddenOutput; // [outputSize][hiddenSize]
        vector<double> dBiasOutput;

public:
    // Constructor
    ForwardNeuralNetwork(int inputSize,
                         int hiddenSize,
                         int outputSize,
                         double learningRate,
                         unique_ptr<ActivationFunction> hiddenActivation,
                         unique_ptr<ActivationFunction> outputActivation);

    // Initialize weights and biases with random values
    void initializeWeights();

    // Forward propagation
    vector<double> forward(const vector<double> &input);

    // Compute Mean Squared Error
    double computeLoss(const vector<double> &predictions, const vector<double> &targets) const;

    // Backward propagation
    void backward(const vector<double> &input, const vector<double> &target);

    // Update weights and biases
    void updateParameters();

    void train(const vector<vector<double> > &trainInputs,
                const vector<vector<double> > &trainTargets,
                const vector<vector<double> > &testInputs,
                const vector<vector<double> > &testTargets,
                int epochs,
                int batchSize,
                vector<double> &totalTrainMAE,
                vector<double> &totalTestMAE);

    vector<double> flatten(const vector<vector<double> > &list);

    vector<double> predict(const vector<vector<double> > &inputs,const vector<vector<double> > &targets);

     void resetGradients();

    // Evaluation Metrics
    double computeMSE(const vector<double> &predictions, const vector<double> &targets) const;
    double computeRMSE(const vector<double> &predictions, const vector<double> &targets) const;
    double computeMAE(const vector<double> &predictions, const vector<double> &targets) const;
    double computeRSquared(const vector<double> &predictions, const vector<double> &targets) const;



    // Getters (optional, for inspection)
    const vector<vector<double> >& getWeightsInputHidden() const { return weightsInputHidden; }
    const vector<double>& getBiasHidden() const { return biasHidden; }
    const vector<vector<double> >& getWeightsHiddenOutput() const { return weightsHiddenOutput; }
    const vector<double>& getBiasOutput() const { return biasOutput; }
    const vector<double>& getActivationHidden() const { return activationHidden; }
    const vector<double>& getActivationOutput() const { return activationOutput; }
    const vector<double>& getZHidden() const { return zHidden; }
    const vector<double>& getZOutput() const { return zOutput; }

    // Destructor
    ~ForwardNeuralNetwork() {}
};
