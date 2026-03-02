#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

// IMPORTANT!! Based on table, my personalized data sets are 32 and 66. 
// First test with sanity data checks 1 and 2.

using namespace std;    

// load_data function borrowed from online sources
// https://www.geeksforgeeks.org/cpp/read-file-using-ifstream-in-cpp/
// https://documents.uow.edu.au/~lukes/textbook/notes-cpp/io/readtextfile.html

void load_data(const string& filename, vector<vector<double>>& data) {
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "Error opening file" << endl; // Terminate
        exit(1);
    }

    string line;

    while (getline(file, line)) {
        stringstream ss(line);

        vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        data.push_back(row);

        // Our data parameter is a vector of vectors, each "row" is an instance with its features. 
    }

    file.close();
}

// Used chatgpt to generate a random function before we implement k-cross validation
// Ommitted as we implement k-cross validation

// double random_accuracy() {
//     static random_device rd;
//     static mt19937 gen(rd());
//     static uniform_real_distribution<double> dist(0.0, 1.0);

//     return dist(gen);
// }

// The leave_one_out k-cross validation function is heavily based on the MATLAB pseudocode from Prof Keogh's slides.
// Various online sources were used to translate to C++

double leave_one_out(const vector<vector<double>>& data, const vector<int>& current_features) {
    int number_correctly_classified = 0;

    for (int i = 0; i < data.size(); i++) {
        double nearest_distance = numeric_limits<double>::infinity();
        int nearest_label = -1;

        for (int k = 0; k < data.size(); k++) {

            if (k == i) // Do not compare an instance with itself
                continue;

            double distance = 0.0;

            // Compute distance using the features in the current set we have considered
            // https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/
            // Used this resource to correctly calculate Euclidean distance in C++
            for (int feature : current_features) {

                // We do +1 because feature 0 is the classification
                double diff = data[i][feature + 1] - data[k][feature + 1];

                distance += diff * diff;
            }

            distance = sqrt(distance);

            if (distance < nearest_distance) {
                nearest_distance = distance; // Update the nearest distance
                nearest_label = data[k][0];  // label stored here at index 0
            }
        }

        if (data[i][0] == nearest_label)
            number_correctly_classified++; // If the calculated nearest neighbor is of the same class, we classified correctly!
    }

    return static_cast<double>(number_correctly_classified) / data.size(); // Calulate accuracy
}

int main() {
    vector<vector<double>> data;          
    string filename; 
    string userInput;
    cout << "--- Welcome to the C++ Feature Selection Algorithm --- " << endl;
    cout << "--- Enter the type of algorithm you'd like to use --- " << endl;
    cout << "Forward Selection (Type 1 and Enter)" << endl;
    cout << "Backwards Elimination (Type 2 and Enter)" << endl;
    
    cin >> userInput;

    cout << "--- Enter the name of the file you'd like to use --- " << endl;
    cin >> filename;

    // IMPORTANT!!!! The code below was written to follow Professor Keogh's pseudocode almost exactly. The forward selection algorithm
    // is implemented below. Most of the variables follow the pseudocode of the slides, and I tried my best to maintain the struture
    // as it was presented in the instructions. (Some small changes were made rewriting this code in C++)

    if (userInput == "1") {
        load_data(filename, data);

        int num_features = data[0].size() - 1;
        cout << "Loaded " << data.size() << " instances" << endl;
        cout << "Each has " << num_features << " features" << endl << endl;
 
        vector<int> current_set; // current_set will be populated as we add more and more features. For now, it is empty
        double global_best_accuracy = 0.0;
        vector<int> global_best_set;

        // The load_data function creates a vector of vectors in which the first index of a "row" is the classification.
        // As such, we add 1 to our iterator "i" to correctly identify the feature. 

        for (int i = 0; i < num_features; i++) {
            cout << "On level " << i+1 << " of the search tree" << endl;

            int best_feature = -1;
            double best_accuracy = 0;

            for (int k = 0; k < num_features; k++) {

                if (find(current_set.begin(), current_set.end(), k) != current_set.end()) {
                    continue; 
                    // https://www.geeksforgeeks.org/cpp/string-find-in-cpp/
                    // Used std::find in the string library to ignore features that already have been added
                }

                cout << "-- Considering adding feature " << k + 1 << endl;
                
                // Here the code deviates from the pseudocode. Instead of having a "feature_to_add" parameter in the leave_one_out
                // function, we create a "trial_set" that acts as the current set with the feature included so we don't 
                // have to delete the feature from the "current_set" to test other features. 
                // We only want to include the best feature every round, so current_set is untouched until all are tested. 
                vector<int> trial_set = current_set;
                trial_set.push_back(k);
                double accuracy = leave_one_out(data, trial_set);

                if (accuracy > best_accuracy) {
                    best_accuracy = accuracy;
                    best_feature = k;
                }
            }

            if (best_feature == -1) {
                break;            
            }

            current_set.push_back(best_feature); // Push back the best feature this round.

            // Keep track of best accuracy globally, so we can have the relevant subset and their accuracy when the program finishes
            if (best_accuracy > global_best_accuracy) {
                global_best_accuracy = best_accuracy;
                global_best_set = current_set;
            }

            cout << "On level " << i+1 << " I added feature " << best_feature+1 << " with accuracy " << best_accuracy << endl << endl;
        }

        cout << "Final Total Features: ";

        for (int feature : current_set) {
            cout << feature + 1 << " " << endl;
        }

        cout << endl;

        cout << endl << "==== GLOBAL BEST ACCURACY ====" << endl;
        cout << "Best Accuracy: " << global_best_accuracy << endl;
        cout << "Best Feature Set: ";

        for (int feature : global_best_set) {
            cout << feature + 1 << " ";
        }

        cout << endl;
    }

    return 0;
}