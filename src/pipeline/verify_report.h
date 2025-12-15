#pragma once

#include <string>

namespace naja::pipeline {

struct CsvShape {
    int rows = 0;
    int cols = 0;
};

CsvShape csv_shape(const std::string& path);
int text_line_count(const std::string& path);

} // namespace naja::pipeline


