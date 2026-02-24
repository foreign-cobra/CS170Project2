# CS170Project2
This is a Feature Selection Algorithm that uses either forward selection or backwards elimination, paired with a nearest neighbor classification algorithm to determine which set of features in a data set are the most relevant for classification. 

## Algorithms Implemented

1. **Forward Selection**
2. **Backward Elimination**

## Evaluation Method

- Leave-One-Out Cross Validation
- Nearest Neighbor (with Euclidean Distance Classification)

## Requirements

- `g++`
- Git

## Installation & Usage

Clone the repository, change your directory, then compile and run:

```bash
git clone https://github.com/foreign-cobra/CS170Project2.git
cd 170Project2
g++ -o feature_selection NearestNeighbor.cpp
./feature_selection      # Or feature_selection.exe if you are on Windows
