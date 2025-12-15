// Minimal CSV loader using Eigen
#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace csv {

inline Eigen::MatrixXd loadMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    if (data.empty()) {
        throw std::runtime_error("Empty CSV file: " + filename);
    }
    
    int rows = data.size();
    int cols = data[0].size();
    
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }
    
    return mat;
}

inline Eigen::VectorXd loadVector(const std::string& filename) {
    Eigen::MatrixXd mat = loadMatrix(filename);
    if (mat.cols() == 1) {
        return mat.col(0);
    } else if (mat.rows() == 1) {
        return mat.row(0).transpose();
    } else {
        return Eigen::Map<Eigen::VectorXd>(mat.data(), mat.size());
    }
}

} // namespace csv
