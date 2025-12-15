#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace npy {

// Write Eigen matrix to .npy format (C-contiguous, float64)
inline void save(const std::string& filename, const Eigen::MatrixXd& mat) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write file: " + filename);
    }
    
    // Magic string
    file.write("\x93NUMPY", 6);
    
    // Version 1.0
    file.put(0x01);
    file.put(0x00);
    
    // Create header dict
    std::ostringstream header;
    header << "{'descr': '<f8', 'fortran_order': False, 'shape': (";
    header << mat.rows() << ", " << mat.cols() << "), }";
    
    // Pad header to multiple of 64 bytes (including length prefix)
    std::string header_str = header.str();
    size_t total_header_len = 6 + 2 + 2 + header_str.size();  // magic + version + len + dict
    size_t padding = (64 - (total_header_len % 64)) % 64;
    header_str.append(padding, ' ');
    header_str.push_back('\n');
    
    // Write header length (little-endian uint16)
    uint16_t header_len = header_str.size();
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    
    // Write header
    file.write(header_str.c_str(), header_len);
    
    // Write data in row-major (C) order
    // Eigen is column-major by default. Copy to RowMajor temp then write block.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_row = mat;
    file.write(reinterpret_cast<const char*>(mat_row.data()), mat.size() * sizeof(double));
    
    file.close();
}

// Write a 1D uint8 array to .npy (C-contiguous, shape (N,))
inline void save_u8_1d(const std::string& filename, const std::vector<uint8_t>& v) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write file: " + filename);
    }

    file.write("\x93NUMPY", 6);
    file.put(0x01);
    file.put(0x00);

    std::ostringstream header;
    header << "{'descr': '|u1', 'fortran_order': False, 'shape': (";
    header << v.size() << ",), }";

    std::string header_str = header.str();
    size_t total_header_len = 6 + 2 + 2 + header_str.size();
    size_t padding = (64 - (total_header_len % 64)) % 64;
    header_str.append(padding, ' ');
    header_str.push_back('\n');

    uint16_t header_len = header_str.size();
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_str.c_str(), header_len);

    if (!v.empty()) {
        file.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
    }
    file.close();
}

// Load NPY file.
// OPTIMIZATION: Returns the matrix transposed (cols, rows) relative to file shape!
// File shape (R, C) C-order -> Eigen Matrix (C, R) Col-order.
// This avoids data shuffling and is essentially free.
// Caller must interpret dimensions accordingly.
inline Eigen::MatrixXd load_transposed(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open " + filename);
    
    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("Invalid NPY file");
    
    char version[2];
    file.read(version, 2);
    
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    
    std::string header(header_len, ' ');
    file.read(&header[0], header_len);
    
    // Parse shape
    size_t shape_pos = header.find("'shape':");
    if (shape_pos == std::string::npos) throw std::runtime_error("Missing shape in NPY header");
    
    size_t start = header.find('(', shape_pos);
    size_t end = header.find(')', start);
    std::string shape_str = header.substr(start + 1, end - start - 1);
    
    std::replace(shape_str.begin(), shape_str.end(), ',', ' ');
    std::stringstream ss(shape_str);
    int rows, cols;
    ss >> rows >> cols;
    
    // Check fortran_order
    bool fortran_order = header.find("'fortran_order': True") != std::string::npos;
    
    if (fortran_order) {
        // File is Col-major (R, C). Read into (R, C) Col-major.
        Eigen::MatrixXd mat(rows, cols);
        file.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));
        // We promised transpose? No, this returns original.
        // This case is rare for NPY saved by numpy or us.
        // If we want consistent behavior (Transpose), we should transpose this?
        // For now, assume C-order (default).
        return mat.transpose(); 
    } else {
        // File is Row-major (R, C).
        // Data in file: Row 0, Row 1...
        // Read into Eigen Matrix (C, R) Col-major.
        // Eigen Col-major stores Col 0, Col 1...
        // So mat(c, r) stores data[c + r*C].
        // File data[k] = data[c + r*C].
        // This corresponds to mat(c, r) = file[r, c].
        // So the matrix we get is the Transpose of the file matrix.
        
        Eigen::MatrixXd mat(cols, rows);
        file.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));
        return mat;
    }
}

// Standard load (slow if C-order) - kept for compatibility if needed, 
// but we prefer load_transposed for speed.
inline Eigen::MatrixXd load(const std::string& filename) {
    // For coloring tool, we use load_transposed logic manually or call it.
    // If user calls load(), they expect (R, C).
    // We can implement load() by calling load_transposed() and transposing back.
    // Transposing in memory is fast (1s for 3GB).
    return load_transposed(filename).transpose();
}

} // namespace npy
