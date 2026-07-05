#include "pipeline/model_io.h"

#include <stdexcept>

#include "csv_loader.h"
#include "npz.h"
#include "utils.h"

namespace naja::pipeline {

namespace {
std::string bundle_path_for(const std::string& rounding_dir) {
    return rounding_dir + "/polytope.npz";
}
std::string extra_bundle_path_for(const std::string& rounding_dir) {
    return rounding_dir + "/extra.npz";
}
std::string start_sidecar_for(const std::string& rounding_dir) {
    return rounding_dir + "/start.npy";
}
} // namespace

bool rounding_is_bundle(const std::string& rounding_dir, const std::string& /*model_name*/) {
    return path_exists(bundle_path_for(rounding_dir));
}

RoundingReader::RoundingReader(std::string rounding_dir, std::string model_name)
    : rounding_dir_(std::move(rounding_dir)),
      model_name_(std::move(model_name)) {
    legacy_prefix_ = rounding_dir_ + "/" + model_name_ + "_rounding";
    bundle_path_ = bundle_path_for(rounding_dir_);
    extra_bundle_path_ = extra_bundle_path_for(rounding_dir_);
    start_sidecar_ = start_sidecar_for(rounding_dir_);
    layout_ = path_exists(bundle_path_) ? RoundingLayout::Bundle : RoundingLayout::Legacy;
}

// Out-of-line dtor so the unique_ptr<NpzArchive> can use the forward decl in
// the header (NpzArchive is only a complete type in this translation unit).
RoundingReader::~RoundingReader() = default;

npz::NpzArchive& RoundingReader::bundle_archive() {
    if (!bundle_) {
        bundle_ = std::make_unique<npz::NpzArchive>(bundle_path_);
    }
    return *bundle_;
}

npz::NpzArchive* RoundingReader::extra_archive() {
    if (!extra_opened_) {
        extra_opened_ = true;
        if (path_exists(extra_bundle_path_)) {
            extra_ = std::make_unique<npz::NpzArchive>(extra_bundle_path_);
        }
    }
    return extra_.get();
}

Eigen::MatrixXd RoundingReader::A() {
    if (layout_ == RoundingLayout::Bundle) return bundle_archive().matrix("A");
    return csv::loadMatrix(legacy_prefix_ + "_A.csv");
}

Eigen::VectorXd RoundingReader::b() {
    if (layout_ == RoundingLayout::Bundle) return bundle_archive().vector("b");
    return csv::loadVector(legacy_prefix_ + "_b.csv");
}

Eigen::VectorXd RoundingReader::start() {
    // A per-model start sidecar (written by feasible-start recompute) always
    // wins, regardless of layout, so it can override a shared bundle start.
    if (path_exists(start_sidecar_)) {
        return npz::load_npy_vector(start_sidecar_);
    }
    if (layout_ == RoundingLayout::Bundle) return bundle_archive().vector("start");
    return csv::loadVector(legacy_prefix_ + "_start.csv");
}

bool RoundingReader::has_backmap() {
    if (layout_ == RoundingLayout::Bundle) {
        auto& z = bundle_archive();
        return z.has("T") && z.has("shift");
    }
    return path_exists(legacy_prefix_ + "_T.csv") &&
           path_exists(legacy_prefix_ + "_shift.csv");
}

Eigen::MatrixXd RoundingReader::T() {
    if (layout_ == RoundingLayout::Bundle) return bundle_archive().matrix("T");
    return csv::loadMatrix(legacy_prefix_ + "_T.csv");
}

Eigen::VectorXd RoundingReader::shift() {
    if (layout_ == RoundingLayout::Bundle) return bundle_archive().vector("shift");
    return csv::loadVector(legacy_prefix_ + "_shift.csv");
}

RoundingReader::ExtraSource RoundingReader::resolve_extra() {
    if (extra_source_ != ExtraSource::Unknown) return extra_source_;

    // Prefer a bundled extra.npz, but fall back to legacy CSVs even in bundle
    // layout: `prepare` writes per-model KO deltas as CSVs, and those must be
    // honored alongside an inherited (bundle) polytope.npz.
    npz::NpzArchive* z = extra_archive();
    if (z) {
        bool a = z->has("extra_A");
        bool b = z->has("extra_b");
        if (a != b) {
            throw std::runtime_error("extra.npz must contain both extra_A and extra_b: " + extra_bundle_path_);
        }
        if (a && b) { extra_source_ = ExtraSource::Bundle; return extra_source_; }
    }

    bool a = path_exists(legacy_prefix_ + "_extra_A.csv");
    bool b = path_exists(legacy_prefix_ + "_extra_b.csv");
    if (a != b) {
        throw std::runtime_error("extra constraints must be both-present or both-absent: " +
                                 legacy_prefix_ + "_extra_A.csv / " + legacy_prefix_ + "_extra_b.csv");
    }
    extra_source_ = (a && b) ? ExtraSource::Legacy : ExtraSource::None;
    return extra_source_;
}

bool RoundingReader::has_extra() {
    return resolve_extra() != ExtraSource::None;
}

Eigen::MatrixXd RoundingReader::extra_A() {
    switch (resolve_extra()) {
        case ExtraSource::Bundle: return extra_archive()->matrix("extra_A");
        case ExtraSource::Legacy: return csv::loadMatrix(legacy_prefix_ + "_extra_A.csv");
        default: throw std::runtime_error("model_io: no extra constraints for " + model_name_);
    }
}

Eigen::VectorXd RoundingReader::extra_b() {
    switch (resolve_extra()) {
        case ExtraSource::Bundle: return extra_archive()->vector("extra_b");
        case ExtraSource::Legacy: return csv::loadVector(legacy_prefix_ + "_extra_b.csv");
        default: throw std::runtime_error("model_io: no extra constraints for " + model_name_);
    }
}

std::string RoundingReader::start_write_path() const {
    if (layout_ == RoundingLayout::Bundle) return start_sidecar_;
    return legacy_prefix_ + "_start.csv";
}

} // namespace naja::pipeline
