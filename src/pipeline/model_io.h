#pragma once
// Rounding-contract loader that abstracts over two on-disk layouts:
//
//   Bundle (preferred):
//     rounding/polytope.npz   arrays: A, b, start, T, shift  (shared base rounding;
//                             one file, symlinkable wholesale for KO inheritance)
//     rounding/extra.npz      arrays: extra_A, extra_b        (per-model KO delta, optional)
//     rounding/start.npy      per-model recomputed feasible start (sidecar, optional;
//                             overrides the bundle's start without mutating the shared npz)
//     rounding/manifest.json  metadata (optional, informational)
//
//   Legacy (fallback):
//     rounding/<model>_rounding_A.csv, _b.csv, _start.csv, _T.csv, _shift.csv,
//     rounding/<model>_rounding_extra_A.csv, _extra_b.csv
//
// Layout is chosen by presence of polytope.npz. Loaders throw std::runtime_error
// on any structural problem. Opened archives are cached so A/b/start/T/shift do
// not re-read and re-parse the (large) bundle repeatedly.

#include <Eigen/Dense>

#include <memory>
#include <string>

namespace npz { class NpzArchive; }

namespace naja::pipeline {

enum class RoundingLayout { Bundle, Legacy };

// True if a bundle (polytope.npz) exists for this model. Cheap: existence check
// only, so callers that must not depend on npz parsing can use it.
bool rounding_is_bundle(const std::string& rounding_dir, const std::string& model_name);

class RoundingReader {
public:
    RoundingReader(std::string rounding_dir, std::string model_name);
    ~RoundingReader();

    RoundingLayout layout() const { return layout_; }
    bool is_bundle() const { return layout_ == RoundingLayout::Bundle; }

    Eigen::MatrixXd A();
    Eigen::VectorXd b();
    Eigen::VectorXd start();   // start.npy sidecar (if present) overrides bundle/legacy start

    bool has_backmap();
    Eigen::MatrixXd T();
    Eigen::VectorXd shift();

    bool has_extra();
    Eigen::MatrixXd extra_A();
    Eigen::VectorXd extra_b();

    // Where a recomputed per-model feasible start should be written for this
    // layout (start.npy for bundles, <prefix>_start.csv for legacy).
    std::string start_write_path() const;

private:
    std::string rounding_dir_;
    std::string model_name_;
    std::string legacy_prefix_;        // rounding_dir/<model>_rounding
    std::string bundle_path_;          // rounding/polytope.npz
    std::string extra_bundle_path_;    // rounding/extra.npz
    std::string start_sidecar_;        // rounding/start.npy
    RoundingLayout layout_;

    std::unique_ptr<npz::NpzArchive> bundle_;
    std::unique_ptr<npz::NpzArchive> extra_;
    bool extra_opened_ = false;

    // Where the KO delta lives: an extra.npz bundle, legacy CSVs, or nowhere.
    // Resolved once, lazily, since a bundle polytope may coexist with legacy
    // per-model extra CSVs written by `prepare`.
    enum class ExtraSource { Unknown, None, Bundle, Legacy };
    ExtraSource extra_source_ = ExtraSource::Unknown;
    ExtraSource resolve_extra();

    npz::NpzArchive& bundle_archive();
    npz::NpzArchive* extra_archive();  // nullptr if no extra bundle
};

} // namespace naja::pipeline
