#pragma once // Ensures the file is included only once

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <map>
#include <stdexcept>
#include <random>

using namespace std; // Use the standard namespace

// Namespace to encapsulate the preprocessing functions
namespace DataPreprocessor {

    // Function to read CSV data and store it
    vector<vector<string> > readCSV(const string& file_path);

    // A simple trim function to remove leading/trailing whitespace
    static inline string trim(const string &s);

    // Main function to remove BOM, trim cells, and print the "head" of the data
    vector<string> processAndClean(vector<vector<string> >& data);

    // Declares the non-template function for strings
    void printHead(const vector<vector<string> >& data, const vector<string>& labels, size_t num_rows, int col_width);

    // Templated function for numeric types
    template <typename T>
    void printHead(const vector<vector<T> >& data, const vector<string>& labels, size_t num_rows, int col_width) {
        // 1. Print the labels
        for (const auto& label : labels) {
            printf("%-*s", col_width, label.c_str());
        }
        printf("\n");

        // 2. Determine how many rows to print
        size_t rows_to_print = min(num_rows, data.size());

        // 3. Print the data rows
        for (size_t i = 0; i < rows_to_print; ++i) {
            for (const auto& value : data[i]) {
                printf("%-*f", col_width, static_cast<double>(value)); // Print numerical values
            }
            printf("\n");
        }
    }

    //helper function, obvious
    bool isNumeric(const string& str);

    // Helper function to check if a string is numeric, returns cells that are supposed to be for faster normalization
    vector<int> getNumericIndex(vector<string>& row, vector<int>& nonNumerics);

    // Function to normalize numerical data (min-max scaling)
    void normalize(vector<double>& data, double min_value, double max_value);

    // Function to standardize numerical data (z-score normalization), default
    void standardize(vector<vector<string> >& data, const vector<int>& numericIndeces);

    // Function to one-hot encode categorical data
    void oneHotEncode(vector<vector<string> >& data, vector<int>& nonNumerics, vector<string>& labels);

    template<typename T>
    vector<vector<T> > convertData(const vector<vector<string> >& data, const string& type) {
        vector<vector<T> > converted_data(data.size()); // New vector to hold converted data

        // Function pointer for conversion
        auto convertTo = [](const string& value, const string& type) -> T {
            try {
                if (type == "int") {
                    return static_cast<T>(stoi(value));
                } else if (type == "double") {
                    return static_cast<T>(stod(value));
                } else if (type == "long") {
                    return static_cast<T>(stol(value));
                } else {
                    throw invalid_argument("Unsupported type: " + type);
                }
            } catch (const invalid_argument& e) {
                cerr << "Invalid conversion for value: " << value << " - Skipping..." << endl;
                return static_cast<T>(0); // Default value for invalid input
            } catch (const out_of_range& e) {
                cerr << "Value out of range for conversion: " << value << " - Skipping..." << endl;
                return static_cast<T>(0); // Default value for out-of-range input
            }
        };

        // Iterate through the dataset and apply the conversion
        for (size_t i = 0; i < data.size(); ++i) {
            for (const auto& value : data[i]) {
                converted_data[i].push_back(convertTo(value, type));
            }
        }

        return converted_data;
    }

    template<typename T>
    void shuffleData(vector<vector<T> >& data) {

        cout << "Shuffling data\n";
        // Ensure data is not empty
        if (data.empty()) {
            throw invalid_argument("Data must not be empty.");
        }

        // Create a vector of indices corresponding to the rows
        vector<size_t> indices(data.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        // Randomly shuffle the row indices
        random_device rd;
        mt19937 gen(rd());
        shuffle(indices.begin(), indices.end(), gen);

        // Create shuffled version of data
        vector<vector<T> > shuffled_data(data.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            shuffled_data[i] = std::move(data[indices[i]]);
        }

        // Replace original data with shuffled version
        data = std::move(shuffled_data);
    }

    // Function to split dataset into training and testing sets
    template<typename T>
    void splitData(const vector<vector<T> >& data,
                   double train_ratio,
                   vector<vector<T> >& train_data,
                   vector<vector<T> >& test_data) {

    if(train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw invalid_argument("train_ratio must be between 0.0 and 1.0");
        }

    cout << "\nSplitting Data: (" << train_ratio << "/" << 1 - train_ratio << ")\n";

        size_t total_rows = data.size();
        size_t train_size = static_cast<size_t>(total_rows * train_ratio);
        train_data.assign(data.begin(), data.begin() + train_size);
        test_data.assign(data.begin() + train_size, data.end());
    }

}
