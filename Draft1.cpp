#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <random>

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

double random_accuracy() {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<double> dist(0.0, 1.0);

    return dist(gen);
}

// For now, as suggested in Prof Keogh's slides, we simply return a random accuracy to focus on the forward selection algorithm
double leave_one_out(const vector<vector<double>>& data, const vector<int>& current_features) {
    return random_accuracy();
}

int main() {
    vector<vector<double>> data;          
    string filename; 

    cout << "--- Welcome to the C++ Feature Selection Algorithm --- " << endl;
    cout << "--- Enter the name of the file you'd like to use --- " << endl;

    cin >> filename;

    load_data(filename, data);

    return 0;
}