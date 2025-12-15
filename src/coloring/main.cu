#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <future>
#include <deque>
#include <numeric>

#include "npy.h"
#include "soft_kernels.cuh"
#include "device_utils.h"
#include "utils.h"
#include "coloring_io.h"

using coloring::ModelInfo;
using coloring::Target;

void compute_medians_row(
    const Eigen::MatrixXd& samples, // (n_samples, dims)
    const std::vector<int>& indices,
    Eigen::VectorXd& medians
) {
    int n_targets = indices.size();
    medians.resize(n_targets);
    
    for (int i = 0; i < n_targets; ++i) {
        int col_idx = indices[i];
        if (col_idx < 0 || col_idx >= samples.cols()) {
            medians[i] = 0.0;
            continue;
        }
        
        std::vector<double> vals(samples.rows());
        for (int r = 0; r < samples.rows(); ++r) {
            vals[r] = std::abs(samples(r, col_idx));
        }
        
        if (vals.empty()) {
            medians[i] = 0.0;
        } else {
            size_t n = vals.size() / 2;
            std::nth_element(vals.begin(), vals.begin() + n, vals.end());
            medians[i] = vals[n];
        }
    }
}

int main(int argc, char** argv) {
    try {
        std::string base_dir = ".";
        std::string out_dir_arg;
        
        if (argc > 1) {
            out_dir_arg = argv[1];
            if (!out_dir_arg.empty() && out_dir_arg.back() == '/') out_dir_arg.pop_back();
        }
        
        std::string config_path = base_dir + "/coloring_config.txt";
        auto config = coloring::load_config_map(config_path);
        
        std::string results_dir = out_dir_arg.empty() ? (base_dir + "/experiment/coloring/results") : out_dir_arg;
        std::string out_dir = coloring::get_config_value(config, "OUT_DIR", results_dir);
        
        std::string rxn_map_file = coloring::get_config_value(config, "REACTION_MAP", results_dir + "/reaction_map.csv");
        // We always overwrite/compute sigmas now, but we use this path to save them.
        std::string rxn_sigmas_file = coloring::get_config_value(config, "REACTION_SIGMAS", results_dir + "/reaction_sigmas.csv");
        std::string models_list_file = coloring::get_config_value(config, "MODELS_LIST", results_dir + "/models_list.csv");
        
        ensure_dir(out_dir);
        
        std::cout << "Loading metadata..." << std::endl;
        auto rxn_map = coloring::load_reaction_map(rxn_map_file);
        auto models = coloring::load_models(models_list_file);

        // FORCE compute sigmas
        bool compute_sigmas = true;
        std::vector<Target> targets;
        
        if (compute_sigmas) {
            std::cout << "Computing sigmas (Pass 1) from samples..." << std::endl;
            
            std::vector<std::string> rxn_names = coloring::ordered_reactions(rxn_map);
            std::vector<int> rxn_indices;
            for (const auto& name : rxn_names) {
                if (!name.empty()) rxn_indices.push_back(rxn_map.at(name));
            }
            
            int n_rxns = rxn_names.size();
            std::vector<std::vector<double>> all_medians; 
            all_medians.resize(models.size());
            
            int lookahead = 8;
            std::deque<std::future<Eigen::MatrixXd>> prefetch_futures;
            
            auto load_task = [&](int idx) -> Eigen::MatrixXd {
                try {
                    return npy::load_transposed(models[idx].path);
                } catch (const std::exception& e) {
                    std::cerr << "Failed to load " << models[idx].name << ": " << e.what() << std::endl;
                    return Eigen::MatrixXd(0, 0);
                }
            };
            
            std::cout << "Prefetching for Sigma computation..." << std::endl;
            for (size_t k = 0; k < std::min(models.size(), (size_t)lookahead); ++k) {
                prefetch_futures.push_back(std::async(std::launch::async, load_task, k));
            }
            
            size_t next_to_fetch = lookahead;
            
            for (size_t i = 0; i < models.size(); ++i) {
                if (i % 100 == 0) std::cout << "[" << (i+1) << "/" << models.size() << "] Computing scales for " << models[i].name << "..." << std::endl;
                
                Eigen::MatrixXd samples = prefetch_futures.front().get();
                prefetch_futures.pop_front();
                
                if (next_to_fetch < models.size()) {
                    prefetch_futures.push_back(std::async(std::launch::async, load_task, next_to_fetch));
                    next_to_fetch++;
                }
                
                if (samples.rows() == 0) continue;
                
                Eigen::VectorXd m_medians;
                std::vector<int> indices(samples.cols());
                std::iota(indices.begin(), indices.end(), 0);
                
                compute_medians_row(samples, indices, m_medians);
                
                std::vector<double> vec_medians(m_medians.data(), m_medians.data() + m_medians.size());
                all_medians[i] = vec_medians;
            }
            
            std::cout << "Aggregating scales..." << std::endl;
            std::vector<double> final_sigmas(n_rxns, 0.0);
            
            for (int j = 0; j < n_rxns; ++j) {
                std::vector<double> col_vals;
                for (size_t i = 0; i < models.size(); ++i) {
                    if (j < all_medians[i].size()) {
                        col_vals.push_back(all_medians[i][j]);
                    }
                }
                if (col_vals.empty()) {
                    final_sigmas[j] = 1.0; 
                } else {
                    size_t n = col_vals.size() / 2;
                    std::nth_element(col_vals.begin(), col_vals.begin() + n, col_vals.end());
                    final_sigmas[j] = col_vals[n];
                }
            }
            
            std::ofstream fout(rxn_sigmas_file);
            fout << "Reaction,Sigma\n";
            for (int j = 0; j < n_rxns; ++j) {
                if (!rxn_names[j].empty()) {
                    fout << rxn_names[j] << "," << final_sigmas[j] << "\n";
                }
            }
            fout.close();
            std::cout << "Saved computed sigmas to " << rxn_sigmas_file << std::endl;
            
            targets = coloring::load_targets(rxn_sigmas_file, rxn_map);
        }
        
        if (targets.empty() || models.empty()) {
            std::cerr << "No targets or models found." << std::endl;
            return 1;
        }
        
        std::cout << "Loaded " << targets.size() << " targets and " << models.size() << " models." << std::endl;
        
        int n_targets = targets.size();
        std::vector<int> h_target_indices(n_targets);
        std::vector<double> h_target_sigmas(n_targets);
        
        for (int i = 0; i < n_targets; ++i) {
            h_target_indices[i] = targets[i].index;
            h_target_sigmas[i] = targets[i].sigma;
        }
        
        Eigen::VectorXi eig_indices = Eigen::Map<Eigen::VectorXi>(h_target_indices.data(), n_targets);
        Eigen::VectorXd eig_sigmas = Eigen::Map<Eigen::VectorXd>(h_target_sigmas.data(), n_targets);
        
        naja::gpu::DVector<int> d_target_indices(eig_indices);
        naja::gpu::DVector<double> d_target_sigmas(eig_sigmas);
        naja::gpu::DVector<double> d_row_O(n_targets);
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        Eigen::MatrixXd O(models.size(), targets.size());
        std::ofstream u_file(out_dir + "/universality_scores.csv");
        u_file << "Model,SampleIdx,Universality\n";
        
        int lookahead = 8;
        std::deque<std::future<Eigen::MatrixXd>> prefetch_futures;
        
        auto load_task = [&](int idx) -> Eigen::MatrixXd {
            try {
                return npy::load_transposed(models[idx].path);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load " << models[idx].name << ": " << e.what() << std::endl;
                return Eigen::MatrixXd(0, 0);
            }
        };
        
        std::cout << "Prefetching for Overlap computation..." << std::endl;
        for (size_t k = 0; k < std::min(models.size(), (size_t)lookahead); ++k) {
            prefetch_futures.push_back(std::async(std::launch::async, load_task, k));
        }
        
        size_t next_to_fetch = lookahead;
        
        for (size_t i = 0; i < models.size(); ++i) {
            if (i % 100 == 0) std::cout << "[" << (i+1) << "/" << models.size() << "] Processing " << models[i].name << "..." << std::endl;
            
            Eigen::MatrixXd samples = prefetch_futures.front().get();
            prefetch_futures.pop_front();
            
            if (next_to_fetch < models.size()) {
                prefetch_futures.push_back(std::async(std::launch::async, load_task, next_to_fetch));
                next_to_fetch++;
            }
            
            if (samples.rows() == 0) continue;
            
            int n_samples = samples.rows();
            
            naja::gpu::DMatrix<double> d_samples(samples);
            
            coloring::gpu::compute_soft_overlap_row(
                d_samples,
                d_target_sigmas,
                d_target_indices,
                d_row_O,
                3.0
            );
            
            Eigen::VectorXd row = d_row_O.toHost();
            O.row(i) = row;
            
            naja::gpu::DVector<double> d_u_scores(n_samples);
            coloring::gpu::compute_universality(
                d_samples,
                d_target_sigmas,
                d_target_indices,
                d_u_scores,
                3.0
            );
            
            Eigen::VectorXd u_scores = d_u_scores.toHost();
            
            for (int s = 0; s < n_samples; ++s) {
                u_file << models[i].name << "," << s << "," << u_scores[s] << "\n";
            }
        }
        u_file.close();
        
        std::ofstream out(out_dir + "/soft_overlap_matrix.csv");
        out << ",";
        for (size_t j = 0; j < targets.size(); ++j) {
            out << targets[j].name << (j < targets.size() - 1 ? "," : "");
        }
        out << "\n";
        
        for (size_t i = 0; i < models.size(); ++i) {
            out << models[i].name << ",";
            for (size_t j = 0; j < targets.size(); ++j) {
                out << O(i, j) << (j < targets.size() - 1 ? "," : "");
            }
            out << "\n";
        }
        out.close();
        
        Eigen::VectorXd T = O.colwise().mean();
        std::ofstream t_file(out_dir + "/reaction_tolerance.csv");
        t_file << "Reaction,Tolerance\n";
        for (size_t j = 0; j < targets.size(); ++j) {
            t_file << targets[j].name << "," << T(j) << "\n";
        }
        t_file.close();
        
        std::ofstream cfg_out(out_dir + "/config_used.txt");
        cfg_out << "# Config used for this run\n";
        cfg_out << "OUT_DIR=" << out_dir << "\n";
        cfg_out << "REACTION_MAP=" << rxn_map_file << "\n";
        cfg_out << "REACTION_SIGMAS=" << rxn_sigmas_file << "\n";
        cfg_out << "MODELS_LIST=" << models_list_file << "\n";
        cfg_out.close();
        
        std::cout << "Done. Results saved to " << out_dir << std::endl;
        
        cudaStreamDestroy(stream);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
