#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <stdexcept>
#include "runtime_config.h"

struct ProfileData {
    double load_time = 0.0;
    double cube_center_time = 0.0;
    double barrier_time = 0.0;
    double upload_time = 0.0;
    double sampling_time = 0.0;
    double download_time = 0.0;
    double backtransform_time = 0.0;
    double write_time = 0.0;
    double total_time = 0.0;

    int n_chains = 0;
    int n_samples = 0;
    int reduced_dim = 0;
    int original_dim = 0;
    int constraints = 0;
    int thinning = 0;

    double throughput = 0.0;

    void write_json(const std::string& filename, const RuntimeConfig& cfg) const {
        std::ofstream f(filename);
        if (!f.is_open()) {
            throw std::runtime_error("cannot write profile: " + filename);
        }

        auto now = std::time(nullptr);
        char timestamp[100];
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        f << "{\n";
        f << "  \"timestamp\": \"" << timestamp << "\",\n";
        f << "  \"config\": {\n";
        f << "    \"n_chains\": " << n_chains << ",\n";
        f << "    \"n_samples\": " << n_samples << ",\n";
        f << "    \"reduced_dim\": " << reduced_dim << ",\n";
        f << "    \"original_dim\": " << original_dim << ",\n";
        f << "    \"constraints\": " << constraints << ",\n";
        f << "    \"thinning\": " << thinning << ",\n";
        f << "    \"tpb_ss\": " << cfg.TPB_SS << ",\n";
        f << "    \"gpu_device\": " << cfg.GPU_DEVICE << ",\n";
        f << "    \"back_transform\": " << (cfg.BACK_TRANSFORM ? "true" : "false") << "\n";
        f << "  },\n";
        f << "  \"timing\": {\n";
        f << "    \"load_polytope_s\": " << std::fixed << std::setprecision(4) << load_time << ",\n";
        f << "    \"cube_center_lp_s\": " << cube_center_time << ",\n";
        f << "    \"barrier_rotation_s\": " << barrier_time << ",\n";
        f << "    \"upload_to_gpu_s\": " << upload_time << ",\n";
        f << "    \"gpu_sampling_s\": " << sampling_time << ",\n";
        f << "    \"download_from_gpu_s\": " << download_time << ",\n";
        f << "    \"backtransform_s\": " << backtransform_time << ",\n";
        f << "    \"write_output_s\": " << write_time << ",\n";
        f << "    \"total_s\": " << total_time << "\n";
        f << "  },\n";
        f << "  \"performance\": {\n";
        f << "    \"total_samples\": " << (n_chains * n_samples) << ",\n";
        f << "    \"throughput_samples_per_sec\": " << std::fixed << std::setprecision(1) << throughput << ",\n";
        f << "    \"sampling_fraction\": " << std::fixed << std::setprecision(3) << (sampling_time / total_time) << "\n";
        f << "  }\n";
        f << "}\n";
    }
};


