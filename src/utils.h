#pragma once
#include <string>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <vector>

inline std::string make_absolute_path(const std::string& path) {
    if (path.empty()) {
        return path;
    }
    char resolved[PATH_MAX];
    if (realpath(path.c_str(), resolved)) {
        return std::string(resolved);
    }
    if (path.front() == '/') {
        return path;
    }
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd))) {
        std::string result(cwd);
        result += "/";
        result += path;
        return result;
    }
    return path;
}

inline bool path_exists(const std::string& path) {
    if (path.empty()) return false;
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

inline bool is_directory(const std::string& path) {
    if (path.empty()) return false;
    struct stat st;
    return stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFDIR);
}

inline void ensure_dir(const std::string& dir) {
    if (dir.empty()) {
        return;
    }
    std::string path = dir;
    if (path.size() > 1 && (path.back() == '/' || path.back() == '\\')) {
        path.pop_back();
    }

    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if ((st.st_mode & S_IFDIR) == 0) {
            throw std::runtime_error("Path exists but is not a directory: " + path);
        }
        return;
    }

    auto slash = path.find_last_of("/\\");
    if (slash != std::string::npos) {
        std::string parent = path.substr(0, slash);
        if (!parent.empty()) {
            ensure_dir(parent);
        }
    }

    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + path);
    }
}

inline std::string current_timestamp() {
    auto now = std::time(nullptr);
    char buf[64];
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now))) {
        return std::string(buf);
    }
    return {};
}

inline std::string current_datestamp_compact() {
    auto now = std::time(nullptr);
    char buf[32];
    if (std::strftime(buf, sizeof(buf), "%Y%m%d", std::localtime(&now))) {
        return std::string(buf);
    }
    return {};
}

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};



