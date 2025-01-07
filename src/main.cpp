#include "../include/DataPreprocessing.h"// Include your header file
#include "../include/ForwardNeuralNetwork.h"
#include "../include/ActivationFunction.h"
#include <sciplot/sciplot.hpp> //don't worry about red squiggles
#include <cstdio>
#include <string>

int main() {

    vector<double> trainTargetsFlattened, testTargetsFlattened;
    vector<int> numerics, nonNumerics;
    vector<vector<double> > testSet, trainSet, convertedData;
    vector<string> labels;
    vector<double> testPredictions, trainPredictions;
    string target;
    int targetIndex = -1;

    //path to your CSV file
    string file_path = "./trainingData/insurance.csv";

    // Use the readCSV function to read the file
     vector<vector<string> > data = DataPreprocessor::readCSV(file_path);

    // Check if the data is empty (file not found or error)
    if (data.empty()) {
        cerr << "No data found or failed to read the file." << endl;
        return 1; // Exit with an error code
    }

    //extract labels and get Index of all numerical columns
    cout << "Preview of the CSV data:\n" << endl;
    labels = DataPreprocessor::processAndClean(data); //cleans the csv (removes BOM and whitespace)
    numerics = DataPreprocessor::getNumericIndex(data[1], nonNumerics); //also gets non-numerical cols


    cout << "\nNumerical column Indexes" << endl;
    for(int i = 0; i < (int)numerics.size(); i++) {
      cout << numerics[i] << " ";
    }
    cout << endl;

    cout << "\nnonNumerical column Indexes" << endl;
    for(int i = 0; i < (int)nonNumerics.size(); i++) {
      cout << nonNumerics[i] << " ";
    }
    cout << endl;

    DataPreprocessor::standardize(data, numerics); //standardize numerical data

    DataPreprocessor::printHead(data, labels, 10, 12); //print head

    DataPreprocessor::oneHotEncode(data, nonNumerics, labels); //encode non-numericals

    convertedData = DataPreprocessor::convertData<double>(data, "double"); //convert all columns to doubles

    DataPreprocessor::shuffleData(data); //shuffle to break order bias and improve generalization

    DataPreprocessor::printHead(data, labels, 5, 12);

    DataPreprocessor::splitData(convertedData, 0.8, trainSet, testSet); //split 80/20 training test

    cout << "\n Training set:\n";
    DataPreprocessor::printHead<double>(trainSet, labels, 5, 12);

    cout << "\n Test set:\n";
    DataPreprocessor::printHead<double>(testSet, labels, 5, 12);

    cout << endl;

    cout << "Make sure Data looks good before moving on. Hit enter to continue\n";
    cin.get();

    int numFeatures = trainSet[0].size() - 1;

    while (targetIndex == -1) {
        cout << "Choose training column: ";

        for (const auto &cell : labels) {
            cout << cell << " ";
        }
        cout << endl;

        cin >> target;

        bool found = false;
        for (size_t i = 0; i < labels.size(); ++i) {
            if (target == labels[i]) { // Using == for comparison
                targetIndex = i;
                found = true;
                break;
            }
        }

        if (!found) {
            std::cout << "Invalid column, try again" << std::endl;
        }
    }

    // Prepare training data
    vector<vector<double> > trainInputs;
    vector<vector<double> > trainTargets;
    tie(trainInputs, trainTargets) = DataPreprocessor::prepareInputsAndTargets(trainSet, targetIndex);

    // Prepare testing data
    vector<vector<double> > testInputs;
    vector<vector<double> > testTargets;
    tie(testInputs, testTargets) = DataPreprocessor::prepareInputsAndTargets(testSet, targetIndex);

    // Display separated data
    cout << "\nSeparated Training Inputs and Targets." << endl;
    cout << "Training Inputs Size: " << trainInputs.size() << " x " << trainInputs[0].size() << endl;
    cout << "Training Targets Size: " << trainTargets.size() << " x " << trainTargets[0].size() << endl;

    cout << "Test Inputs Size: " << testInputs.size() << " x " << testInputs[0].size() << endl;
    cout << "Test Targets Size: " << testTargets.size() << " x " << testTargets[0].size() << endl << endl;

    // Pause before proceeding
    cout << "Ready to instantiate and train the Neural Network. Hit enter to continue.";
    cin.get();

    // Define network architecture
    int inputSize = numFeatures;      // Number of input features
    int hiddenSize = 6;              // Number of neurons in hidden layer (adjust as needed)
    int outputSize = 1;               // Single output neuron for binary classification
    double learningRate = 0.0005;       // Learning rate (adjust as needed)

    // choose activation functions
    unique_ptr<ActivationFunction> hiddenActivation = make_unique<ReLU>();
    unique_ptr<ActivationFunction> outputActivation = make_unique<Linear>();

    // construct the neural network
    ForwardNeuralNetwork nn(inputSize,
                             hiddenSize,
                             outputSize,
                             learningRate,
                             std::move(hiddenActivation),
                             std::move(outputActivation));

    // Define training parameters
    int epochs = 500;                // Number of training epochs
    int batchSize = 50;               // batch size

   //for graphing Mean Absolute Error, making sure test and train accuracy is consistant
    vector<double> totalTrainMAE, totalTestMAE;

    // Begin training
    cout << "Starting training..." << endl;
    nn.train(trainInputs, trainTargets, testInputs, testTargets, epochs, batchSize, totalTrainMAE, totalTestMAE);
    cout << "Training completed." << endl;

    // Pause before evaluations
        std::cout << "Ready to evaluate the Neural Network. Hit enter to continue." << endl;
        std::cin.get();

        testPredictions = nn.predict(testInputs);

        for(const auto &input : trainInputs) {
            vector<double> prediction = nn.forward(input);
            trainPredictions.push_back(prediction[0]);
        }

        // Extract actual targets for Mean Average Error
        testTargetsFlattened = nn.flatten(testTargets);


        trainTargetsFlattened = nn.flatten(trainTargets);
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

           plot.size(360, 200);

            // Plot the training MAE (red line)
            plot.drawCurve(x, totalTrainMAE).label("Train MAE").lineColor("red");
            // Plot the testing MAE (blue line)
            plot.drawCurve(x, totalTestMAE).label("Test MAE").lineColor("blue");

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


//implement dimensionality reduction, LR decay, and regularization
