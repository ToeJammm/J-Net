#include "../include/DataPreprocessing.h"



namespace DataPreprocessor {

    vector<vector<string> > readCSV(const string& file_path) { //reads in a CSV file, consists of cells that make up rows
        vector<vector<string> > data;
        ifstream file(file_path);

        if (!file.is_open()) {
            cerr << "Error: Could not open file " << file_path << endl;
            return data;
        }

        string line;
        while (getline(file, line)) {
            vector<string> row;
            stringstream line_stream(line);
            string cell;

            while (getline(line_stream, cell, ',')) {
                row.push_back(cell);
            }

            if (!row.empty()) {
                data.push_back(row);
            }
        }

        file.close();
        return data;
    }


    // A simple trim function to remove leading/trailing whitespace
    static inline string trim(const string &s) {
        if (s.empty()) return s;

        size_t start = 0;
        while (start < s.size() && isspace((unsigned char)s[start])) {
            ++start;
        }

        size_t end = s.size() - 1;
        while (end > start && isspace((unsigned char)s[end])) {
            --end;
        }

        return s.substr(start, end - start + 1);
    }

    // Main function to remove BOM, trim cells, and print the "head" of the data
    vector<string> processAndClean(vector<vector<string> >& data) {

        if (data.empty()) {
                throw std::invalid_argument("Data is empty; cannot process or clean.");
            }


        // 1. Remove BOM and trim all cells
        for (auto& row : data) {
            for (auto& cell : row) {
                // Check for UTF-8 BOM (0xEF, 0xBB, 0xBF) at the start of the string
                if (cell.size() >= 3 &&
                    (unsigned char)cell[0] == 0xEF &&
                    (unsigned char)cell[1] == 0xBB &&
                    (unsigned char)cell[2] == 0xBF)
                {
                    cell.erase(0, 3);
                }

                // Trim leading and trailing whitespace
                cell = trim(cell);
            }
        }


        vector<string> labels = data[0];
        cout << "Original Labels:" << endl;
        for (const auto& cell : data[0]) {
            cout << cell << " ";
        }
        cout << endl;

        data.erase(data.begin()); // Remove the first row from the data


        // 2. Remove rows with empty cells or "n/a"
        data.erase(
            std::remove_if(data.begin(), data.end(),
                [](const vector<string>& row) {
                    return std::any_of(row.begin(), row.end(),
                        [](const string& cell) {
                            return cell.empty() || cell == "n/a";
                        });
                }),
            data.end());
        return labels;
    }

    void printHead(const vector<vector<string> >& data, const vector<string>& labels, size_t num_rows, int col_width) {
        // 1. Print the labels
        for (const auto& label : labels) {
            printf("%-*s", col_width, label.c_str());
        }
        printf("\n");

        // 2. Determine how many rows to print
        size_t rows_to_print = min(num_rows, data.size());

        // 3. Print the data rows
        for (size_t i = 0; i < rows_to_print; ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                printf("%-*s", col_width, data[i][j].c_str());
            }
            printf("\n");
        }
    }

    bool isNumeric(const string& str) {
        if (str.empty()) return false;
        try {
            size_t idx;
            stod(str, &idx);
            return idx == str.size();  // Ensure the entire string was numeric
        } catch (const invalid_argument& e) {
            return false;  // Not numeric
        } catch (const out_of_range& e) {
            return false;  // Number is out of range
        }
    }

    vector<int> getNumericIndex(vector<string>& row, vector<int>& nonNumerics) {
        vector<int> numericCols;

        if (row.empty()) return numericCols; // Return empty if no data

        size_t numCols = row.size();
        for (size_t cell = 0; cell < numCols; cell++) {
            const string& val = row[cell];
            if (!val.empty() && isNumeric(val)) {
                numericCols.push_back(cell); // Add numeric index
            } else {
            // If not numeric, add to nonNumerical category indexes
            nonNumerics.push_back(cell);
            }
        }
        return numericCols; // Contains indices of numeric cells in the row
    }


    void standardize(vector<vector<string> >& data, const vector<int>& numericIndices) {
        for (int col : numericIndices) {
            double sum = 0.0;
            double sumSquared = 0.0;
            int count = 0;

            // Calculate mean and variance (skip the first row)
            for (size_t rowIdx = 0; rowIdx < data.size(); rowIdx++) {
                const string& value = data[rowIdx][col];
                if (!value.empty() && isNumeric(value)) {
                    double numValue = stod(value);
                    sum += numValue;
                    sumSquared += numValue * numValue;
                    count++;
                }
            }

            if (count == 0) {
                cerr << "Warning: No valid numeric data in column " << col << endl;
                continue; // Skip this column
            }

            double mean = sum / count;
            double variance = (sumSquared / count) - (mean * mean);
            double stddev = sqrt(variance);

            // Apply standardization (skip the first row)
            for (size_t rowIdx = 0; rowIdx < data.size(); rowIdx++) {
                string& value = data[rowIdx][col];
                if (!value.empty() && isNumeric(value)) {
                    double numValue = stod(value);
                    double standardized = (numValue - mean) / stddev;
                    value = to_string(standardized);
                }
            }
        }
    }

    void oneHotEncode(vector<vector<string> >& data, vector<int>& nonNumerics, vector<string>& labels) {
        for (size_t k = 0; k < nonNumerics.size();) { // Use a while loop as nonNumerics will be modified
            int catIndex = nonNumerics[k]; // Index of the category to encode

            // Identify unique values in the category
            unordered_set<string> uniqueVals;
            for (size_t i = 0; i < data.size(); ++i) {
                uniqueVals.insert(data[i][catIndex]);
            }

            if (uniqueVals.size() == 1) {
                // Case 1: Single unique value
                cout << "Category '" << labels[catIndex] << "' has only one unique value: '"
                     << *uniqueVals.begin() << "'.\n";
                cout << "Removing the column as it provides no useful information.\n";

                // Remove the column from data
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i].erase(data[i].begin() + catIndex);
                }

                // Remove the label
                labels.erase(labels.begin() + catIndex);

                // Remove the column index from nonNumerics
                nonNumerics.erase(nonNumerics.begin() + k);

                // Adjust remaining indices in nonNumerics
                for (int& index : nonNumerics) {
                    if (index > catIndex) {
                        --index;
                    }
                }

            } else if (uniqueVals.size() == 2) {
                // Case 2: Binary category
                cout << "Category '" << labels[catIndex] << "' is binary. Encoding as 0/1.\n";

                string firstVal = *uniqueVals.begin();
                string secondVal = *(++uniqueVals.begin());

                for (size_t i = 0; i < data.size(); ++i) {
                    data[i][catIndex] = (data[i][catIndex] == firstVal) ? "1" : "0";
                }

                // Remove the column index from nonNumerics
                nonNumerics.erase(nonNumerics.begin() + k);

            } else if (uniqueVals.size() > 2) {
                // Case 3: Non-binary category
                cout << "Category '" << labels[catIndex] << "' is non-binary. Performing one-hot encoding.\n";

                // Create a map for unique values
                map<string, int> valueToColumnIndex;
                for (const string& value : uniqueVals) {
                    valueToColumnIndex[value] = labels.size(); // Next available column
                    labels.push_back(value); // Add new label name
                }

                // Fill the new columns with 0/1
                for (size_t i = 0; i < data.size(); ++i) {
                    for (const auto& pair : valueToColumnIndex) {
                        data[i].push_back(data[i][catIndex] == pair.first ? "1" : "0");
                    }
                }

                // Remove the original column from data
                cout << "Removing original column: " << labels[catIndex] << "\n";
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i].erase(data[i].begin() + catIndex);
                }

                // Remove the original label
                labels.erase(labels.begin() + catIndex);

                // Remove the column index from nonNumerics
                nonNumerics.erase(nonNumerics.begin() + k);

                // Adjust remaining indices in nonNumerics
                for (int& index : nonNumerics) {
                    if (index > catIndex) {
                        --index;
                    }
                }
            } else {
                // If none of the above cases apply, move to the next index
                ++k;
            }
        }
    }


    pair<vector<vector<double> >, vector<vector<double> > > prepareInputsAndTargets(
        const vector<vector<double> > &dataset,
        int targetIndex
    ) {
        {
            vector<vector<double> > inputs;
            vector<vector<double> > targets;

            for (const auto &row : dataset) {
                if (row.size() <= targetIndex) {
                    throw invalid_argument("Row size is smaller than target index.");
                }

                vector<double> input;
                vector<double> target;

                // Reserve space for efficiency
                input.reserve(row.size() - 1);
                target.reserve(1); // Assuming single target column

                for (size_t i = 0; i < row.size(); ++i) {
                    if (i == static_cast<size_t>(targetIndex)) {
                        target.push_back(row[i]);
                    } else {
                        input.push_back(row[i]);
                    }
                }

                inputs.emplace_back(std::move(input));
                targets.emplace_back(std::move(target));
            }

            return {std::move(inputs), std::move(targets)};
        }

    }
}
