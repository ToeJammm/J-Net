#include "DataPreprocessing.h" // Include your header file
#include "ForwardNeuralNetwork.h"
#include "ActivationFunction.h"
#include <sciplot/sciplot.hpp>
#include <iostream>           // For printing output
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>

int main() {
    // Specify the path to your CSV file
    string file_path = "./insurance.csv";

    // Use the readCSV function to read the file
    vector<vector<string> > data = DataPreprocessor::readCSV(file_path);
    vector<int> numerics, nonNumerics;
    vector<vector<double> > testSet, trainSet, convertedData;
    vector<string> labels;
    vector<double> testPredictions, trainPredictions;
    int size = 0;

    // Check if the data is empty (file not found or error)
    if (data.empty()) {
        cerr << "No data found or failed to read the file." << endl;
        return 1; // Exit with an error code
    }

    // Print a preview of the first 5 rows
    cout << "Preview of the CSV data:\n" << endl;
    labels = DataPreprocessor::processAndClean(data); //cleans the csv (removes BOM and whitespace)
    numerics = DataPreprocessor::getNumericIndex(data[1], nonNumerics);


    cout << "\nNumerical column Indexes" << endl;
    for(int i = 0; i < numerics.size(); i++) {
      cout << numerics[i] << " ";
    }
    cout << endl;

    cout << "\nnonNumerical column Indexes" << endl;
    for(int i = 0; i < nonNumerics.size(); i++) {
      cout << nonNumerics[i] << " ";
    }
    cout << endl;

    DataPreprocessor::standardize(data, numerics); //standardize numerical data

    DataPreprocessor::printHead(data, labels, 10, 12);

    DataPreprocessor::oneHotEncode(data, nonNumerics, labels);

    convertedData = DataPreprocessor::convertData<double>(data, "double");

    DataPreprocessor::shuffleData(data);

    DataPreprocessor::printHead(data, labels, 5, 12);

    DataPreprocessor::splitData(convertedData, 0.8, trainSet, testSet);

    cout << "\n Training set:\n";
    DataPreprocessor::printHead<double>(trainSet, labels, 5, 12);

    cout << "\n Test set:\n";
    DataPreprocessor::printHead<double>(testSet, labels, 5, 12);

    cout << endl;

    cout << "Make sure Data looks good before moving on. Hit enter to continue";
    cin.get();

    // Separate inputs and targets for training and testing
    // Assuming the last column is the target
    int numFeatures = trainSet[0].size() - 1;

    vector<vector<double> > trainInputs;
    vector<vector<double> > trainTargets;
    vector<vector<double> > testInputs;
    vector<vector<double> > testTargets;

    for(auto &row : trainSet) {
        vector<double> input(row.begin(), row.begin() + numFeatures);
        vector<double> target(row.begin() + numFeatures, row.end());
        trainInputs.push_back(input);
        trainTargets.push_back(target);
    }

    for(auto &row : testSet) {
        vector<double> input(row.begin(), row.begin() + numFeatures);
        vector<double> target(row.begin() + numFeatures, row.end());
        testInputs.push_back(input);
        testTargets.push_back(target);
    }

    // Display separated data
    cout << "\nSeparated Training Inputs and Targets." << endl;
    cout << "Training Inputs Size: " << trainInputs.size() << " x " << trainInputs[0].size() << endl;
    cout << "Training Targets Size: " << trainTargets.size() << " x " << trainTargets[0].size() << endl;

    cout << "Test Inputs Size: " << testInputs.size() << " x " << testInputs[0].size() << endl;
    cout << "Test Targets Size: " << testTargets.size() << " x " << testTargets[0].size() << endl << endl;

    // Pause before proceeding
    cout << "Ready to instantiate and train the Neural Network. Hit enter to continue.";
    cin.get();

    // Proceed to instantiate and train the neural network here...
    // Define network architecture
    int inputSize = numFeatures;      // Number of input features
    int hiddenSize = 6;              // Number of neurons in hidden layer (adjust as needed)
    int outputSize = 1;               // Single output neuron for binary classification
    double learningRate = 0.01;       // Learning rate (adjust as needed)

    // Instantiate activation functions
    unique_ptr<ActivationFunction> hiddenActivation = make_unique<ReLU>();
    unique_ptr<ActivationFunction> outputActivation = make_unique<Sigmoid>();

    // Instantiate the neural network
    ForwardNeuralNetwork nn(inputSize,
                             hiddenSize,
                             outputSize,
                             learningRate,
                             std::move(hiddenActivation),
                             std::move(outputActivation));

    // Define training parameters
    int epochs = 100;                // Number of training epochs
    int batchSize = 32;               // Mini-batch size

    vector<double> totalTrainMAE, totalTestMAE;

    // Begin training
    cout << "Starting training..." << endl;
    nn.train(trainInputs, trainTargets, testInputs, testTargets, epochs, batchSize, totalTrainMAE, totalTestMAE);
    cout << "Training completed." << endl;

    // Pause before evaluationsx
        std::cout << "Ready to evaluate the Neural Network. Hit enter to continue.";
        std::cin.get();

        // Make predictions on the test set and training set

        for(const auto &input : testInputs) {
            vector<double> prediction = nn.forward(input);
            testPredictions.push_back(prediction[0]); // Assuming single output
        }

        for(const auto &input : trainInputs) {
            vector<double> prediction = nn.forward(input);
            trainPredictions.push_back(prediction[0]);
        }

        // Extract actual targets for Mean Average Error
        vector<double> testTargetsFlattened;
        for(const auto &target : testTargets) {
            testTargetsFlattened.push_back(target[0]); // Assuming single target
        }

        vector<double> trainTargetsFlattened;
        for(const auto &target : trainTargets) {
            trainTargetsFlattened.push_back(target[0]);
        }

        // Compute evaluation metrics on the training set
        double trainMSE = nn.computeMSE(trainPredictions, trainTargetsFlattened);
        double trainRMSE = nn.computeRMSE(trainPredictions, trainTargetsFlattened);
        double trainMAE = nn.computeMAE(trainPredictions, trainTargetsFlattened);
        double trainRSquared = nn.computeRSquared(trainPredictions, trainTargetsFlattened);

        // Compute evaluation metrics on the test set
        double testMSE = nn.computeMSE(testPredictions, testTargetsFlattened);
        double testRMSE = nn.computeRMSE(testPredictions, testTargetsFlattened);
        double testMAE = nn.computeMAE(testPredictions, testTargetsFlattened);
        double testRSquared = nn.computeRSquared(testPredictions, testTargetsFlattened);

        // Display the metrics
        std::cout << "=== Training Metrics ===\n";
        std::cout << "MSE: " << trainMSE << "\n";
        std::cout << "RMSE: " << trainRMSE << "\n";
        std::cout << "MAE: " << trainMAE << "\n";
        std::cout << "R²: " << trainRSquared << "\n\n";

        std::cout << "=== Test Metrics ===\n";
        std::cout << "MSE: " << testMSE << "\n";
        std::cout << "RMSE: " << testRMSE << "\n";
        std::cout << "MAE: " << testMAE << "\n";
        std::cout << "R²: " << testRSquared << "\n\n";

        // Create the X-axis indices from 0..(n-1)
            vector<int> x(totalTrainMAE.size());
            for(size_t i = 0; i < totalTrainMAE.size(); ++i)
                x[i] = (i);

            // Create a 2D plot
           sciplot::Plot2D plot;

            plot.size(2200, 600); // width = 1200px, height = 600px
            // Plot the training MAE (red line)
            plot.drawCurve(x, totalTrainMAE).label("Train MAE").lineColor("red");
            // Plot the testing MAE (blue line)
            plot.drawCurve(x, totalTestMAE).label("Test MAE").lineColor("blue");

            // Label axes and legend
            plot.xlabel("Epoch");
            plot.ylabel("MAE");
            plot.legend().atOutsideRight();           // place legend outside to the right
            plot.legend().title("MAE Curves");        // legend title, optional

            // Create a Figure with our single plot
           sciplot::Figure fig = { { plot } };

            // Create a Canvas to hold the figure
            sciplot::Canvas canvas = { { fig } };

            // Display the plot in a Gnuplot window (needs gnuplot installed)
            canvas.show();

            // save to file:
            // canvas.save("MAE_curves.png");



    return 0;
}
