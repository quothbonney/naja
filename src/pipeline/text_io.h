#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace naja::pipeline {

inline std::string read_all_text(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot read: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline std::vector<std::string> read_nonempty_trimmed_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot read: " + path);
    std::vector<std::string> out;
    std::string line;
    auto is_ws = [](unsigned char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; };
    while (std::getline(f, line)) {
        while (!line.empty() && is_ws((unsigned char)line.front())) line.erase(line.begin());
        while (!line.empty() && is_ws((unsigned char)line.back())) line.pop_back();
        if (line.empty()) continue;
        out.push_back(line);
    }
    if (out.empty()) throw std::runtime_error("empty file: " + path);
    return out;
}

} // namespace naja::pipeline


