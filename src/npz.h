#pragma once
// Minimal .npz (NumPy zip archive) reader.
//
// An .npz file is a ZIP archive whose members are individual .npy arrays,
// one per named entry (entry "A" is stored as "A.npy" inside the zip).
// This reader parses the ZIP central directory, extracts named members
// (stored or deflate-compressed), and decodes the .npy payload into Eigen.
//
// Supported .npy dtypes: float64 ('<f8') and float32 ('<f4', upcast to double).
// Supported shapes: 1-D (n,) and 2-D (r, c). Both C- and Fortran-order.
//
// Read-only: naja consumes bundles produced by tools/pack_model.py. It never
// writes .npz (packing is done in Python).

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace npz {

class NpzArchive {
public:
    // Opens and indexes the archive. Throws std::runtime_error on any
    // malformed structure or I/O failure.
    explicit NpzArchive(const std::string& path);

    // True if an array with this logical name exists (i.e. "<name>.npy").
    bool has(const std::string& name) const;

    // Names of all arrays in the archive (without the ".npy" suffix).
    std::vector<std::string> names() const;

    // Decode a member as a matrix with the .npy file's logical (rows, cols)
    // shape. A 1-D array of length n decodes to (n, 1). Throws if absent.
    Eigen::MatrixXd matrix(const std::string& name) const;

    // Decode a member as a vector. Accepts 1-D (n,) or 2-D with a singleton
    // dimension ((n,1) or (1,n)); any other 2-D shape is flattened in the
    // file's element order. Throws if absent.
    Eigen::VectorXd vector(const std::string& name) const;

private:
    struct Member {
        std::string name;      // logical name, ".npy" suffix stripped
        uint16_t method = 0;   // 0 = stored, 8 = deflate
        uint64_t comp_size = 0;
        uint64_t uncomp_size = 0;
        uint64_t local_header_offset = 0;
    };

    std::string path_;
    std::vector<char> file_bytes_;   // whole archive held in memory
    std::vector<Member> members_;

    const Member* find(const std::string& name) const;
    // Returns the decompressed .npy payload for a member.
    std::vector<char> read_member_bytes(const Member& m) const;
};

// ---- standalone .npy files (used for the per-model start sidecar) ----

// Load a standalone .npy file (float32/float64, 1-D or 2-D) into Eigen.
// Unlike npy.h::load, this decodes 1-D shapes correctly (as (n, 1) / length n).
Eigen::MatrixXd load_npy_matrix(const std::string& path);
Eigen::VectorXd load_npy_vector(const std::string& path);

// Write a 1-D float64 .npy (shape (n,)), readable by numpy and by the loaders
// above. Used to persist a recomputed per-model feasible start without
// mutating a shared (possibly symlinked) polytope.npz.
void save_npy_vector(const std::string& path, const Eigen::VectorXd& v);

} // namespace npz
