#include "parquet_writer.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/arrow/reader.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace parquet_writer {

namespace {

std::string default_column_name(int64_t index) {
    std::ostringstream oss;
    oss << "dim_" << std::setw(4) << std::setfill('0') << index;
    return oss.str();
}

void throw_if_failed(const arrow::Status& status, const std::string& context) {
    if (!status.ok()) {
        throw std::runtime_error(context + ": " + status.ToString());
    }
}

} // namespace

void write_matrix(const std::string& filename, const Eigen::MatrixXd& mat) {
    const int64_t dims = mat.rows();
    const int64_t samples = mat.cols();
    if (dims == 0 || samples == 0) {
        throw std::runtime_error("Cannot write empty matrix to Parquet");
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();

    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(dims);
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(dims);

    for (int64_t d = 0; d < dims; ++d) {
        arrow::FloatBuilder builder(pool);
        throw_if_failed(builder.Reserve(samples),
                        "Failed to reserve Parquet column buffer");
        for (int64_t s = 0; s < samples; ++s) {
            builder.UnsafeAppend(static_cast<float>(mat(d, s)));
        }
        std::shared_ptr<arrow::Array> array;
        throw_if_failed(builder.Finish(&array), "Failed to finalize Arrow array");
        columns.emplace_back(std::move(array));
        fields.emplace_back(
            arrow::field(default_column_name(d), arrow::float32()));
    }

    auto schema = arrow::schema(fields);
    auto table = arrow::Table::Make(schema, columns, samples);

    auto outfile_result = arrow::io::FileOutputStream::Open(filename);
    if (!outfile_result.ok()) {
        throw std::runtime_error("Failed to open Parquet file '" + filename +
                                 "': " + outfile_result.status().ToString());
    }

    auto outfile = *outfile_result;

    throw_if_failed(
        parquet::arrow::WriteTable(*table, pool, outfile, samples),
        "Failed to write Parquet data");
}

Eigen::MatrixXd read_matrix(const std::string& filename) {
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    
    auto infile_result = arrow::io::ReadableFile::Open(filename);
    if (!infile_result.ok()) {
        throw std::runtime_error("Failed to open Parquet file '" + filename + "': " + infile_result.status().ToString());
    }
    
    // Fix: OpenFile returns arrow::Result
    auto reader_result = parquet::arrow::OpenFile(*infile_result, pool);
    if (!reader_result.ok()) {
        throw std::runtime_error("Failed to create Parquet reader: " + reader_result.status().ToString());
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(reader_result).ValueOrDie();
    
    std::shared_ptr<arrow::Table> table;
    throw_if_failed(reader->ReadTable(&table), "Failed to read Parquet table");
    
    if (!table) {
        throw std::runtime_error("Empty table read from " + filename);
    }
    
    int64_t dims = table->num_columns();
    int64_t samples = table->num_rows();
    
    Eigen::MatrixXd mat(dims, samples);
    
    std::shared_ptr<arrow::Table> combined_table;
    auto res = table->CombineChunks(pool);
    if(res.ok()) {
        combined_table = *res;
    } else {
        combined_table = table; 
    }
    
    for (int64_t d = 0; d < dims; ++d) {
        auto chunked_array = combined_table->column(d);
        int64_t current_row = 0;
        for (const auto& array : chunked_array->chunks()) {
            int64_t length = array->length();
            
            if (array->type_id() == arrow::Type::FLOAT) {
                auto float_array = std::static_pointer_cast<arrow::FloatArray>(array);
                for (int64_t i = 0; i < length; ++i) {
                    mat(d, current_row + i) = static_cast<double>(float_array->Value(i));
                }
            } else if (array->type_id() == arrow::Type::DOUBLE) {
                auto double_array = std::static_pointer_cast<arrow::DoubleArray>(array);
                for (int64_t i = 0; i < length; ++i) {
                    mat(d, current_row + i) = double_array->Value(i);
                }
            } else {
                throw std::runtime_error("Unsupported column type in Parquet file (expected float/double)");
            }
            current_row += length;
        }
    }
    
    return mat;
}

} // namespace parquet_writer
