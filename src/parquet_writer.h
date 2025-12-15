#pragma once

#include <Eigen/Dense>
#include <string>

namespace parquet_writer {

// Writes the matrix (rows = dimensions, cols = samples) into a Parquet file
// where each column corresponds to a dimension and each row corresponds to a
// sample. Throws std::runtime_error on failure.
void write_matrix(const std::string& filename, const Eigen::MatrixXd& mat);

// Reads a Parquet file into an Eigen matrix (rows = dimensions, cols = samples).
// Assumes columns in Parquet correspond to dimensions.
// Throws std::runtime_error on failure.
Eigen::MatrixXd read_matrix(const std::string& filename);

} // namespace parquet_writer
