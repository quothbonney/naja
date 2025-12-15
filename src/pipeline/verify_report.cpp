#include "pipeline/verify_report.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include "utils.h"

namespace naja::pipeline {
namespace {

int count_cols_csv_line(const std::string& line) {
    if (line.empty()) return 0;
    int cols = 1;
    for (char c : line) {
        if (c == ',') cols++;
    }
    return cols;
}

} // namespace

CsvShape csv_shape(const std::string& path) {
    if (!path_exists(path)) {
        throw std::runtime_error("missing file: " + path);
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open file: " + path);
    }
    CsvShape s;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (first) {
            s.cols = count_cols_csv_line(line);
            first = false;
        }
        s.rows++;
    }
    if (s.rows == 0 || s.cols == 0) {
        throw std::runtime_error("empty csv: " + path);
    }
    return s;
}

int text_line_count(const std::string& path) {
    if (!path_exists(path)) {
        throw std::runtime_error("missing file: " + path);
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open file: " + path);
    }
    int n = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) n++;
    }
    if (n == 0) {
        throw std::runtime_error("empty file: " + path);
    }
    return n;
}

}


