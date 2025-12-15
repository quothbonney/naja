#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace coloring {

struct Target {
    std::string name;
    int index;
    double sigma;
};

struct ModelInfo {
    std::string name;
    std::string path;
};

inline std::string trim(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = input.find_last_not_of(" \t\r\n");
    return input.substr(start, end - start + 1);
}

inline std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream(s);
    while (std::getline(token_stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

inline std::map<std::string, std::string> load_config_map(const std::string& filename) {
    std::map<std::string, std::string> config;
    std::ifstream file(filename);
    if (!file.is_open()) {
        return config;
    }
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        if (!key.empty()) {
            config[key] = value;
        }
    }
    return config;
}

inline std::string get_config_value(const std::map<std::string, std::string>& config,
                                    const std::string& key,
                                    const std::string& default_value) {
    auto it = config.find(key);
    return it == config.end() ? default_value : it->second;
}

inline std::map<std::string, int> load_reaction_map(const std::string& filename) {
    std::map<std::string, int> mapping;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + filename);
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        auto tokens = split(line, ',');
        if (tokens.size() < 2) {
            continue;
        }
        mapping[tokens[0]] = std::stoi(tokens[1]);
    }
    return mapping;
}

inline std::vector<Target> load_targets(const std::string& filename,
                                        const std::map<std::string, int>& rxn_map) {
    std::vector<Target> targets;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + filename);
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        auto tokens = split(line, ',');
        if (tokens.size() < 2) {
            continue;
        }
        auto it = rxn_map.find(tokens[0]);
        if (it == rxn_map.end()) {
            std::cerr << "Warning: Target " << tokens[0] << " not found in reaction map." << std::endl;
            continue;
        }
        targets.push_back(Target{tokens[0], it->second, std::stod(tokens[1])});
    }
    return targets;
}

inline std::vector<ModelInfo> load_models(const std::string& filename) {
    std::vector<ModelInfo> models;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + filename);
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        auto tokens = split(line, ',');
        if (tokens.size() < 2) {
            continue;
        }
        models.push_back(ModelInfo{tokens[0], tokens[1]});
    }
    return models;
}

inline std::vector<std::string> ordered_reactions(const std::map<std::string, int>& rxn_map) {
    size_t max_index = 0;
    for (const auto& kv : rxn_map) {
        if (kv.second > static_cast<int>(max_index)) {
            max_index = kv.second;
        }
    }
    std::vector<std::string> names(max_index + 1);
    for (const auto& kv : rxn_map) {
        if (kv.second >= 0 && static_cast<size_t>(kv.second) < names.size()) {
            names[kv.second] = kv.first;
        }
    }
    return names;
}

inline std::vector<double> load_sigma_vector(const std::string& filename,
                                             const std::map<std::string, int>& rxn_map,
                                             size_t vector_size) {
    std::vector<double> sigmas(vector_size, 0.0);
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + filename);
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        auto tokens = split(line, ',');
        if (tokens.size() < 2) {
            continue;
        }
        auto it = rxn_map.find(tokens[0]);
        if (it == rxn_map.end()) {
            continue;
        }
        if (static_cast<size_t>(it->second) < sigmas.size()) {
             sigmas[it->second] = std::stod(tokens[1]);
        }
    }
    return sigmas;
}

}  // namespace coloring











