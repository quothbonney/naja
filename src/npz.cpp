#include "npz.h"

#include <zlib.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace npz {
namespace {

// ---- little-endian scalar reads from an in-memory buffer ----

uint16_t rd_u16(const char* p) {
    return static_cast<uint16_t>(static_cast<unsigned char>(p[0])) |
           (static_cast<uint16_t>(static_cast<unsigned char>(p[1])) << 8);
}

uint32_t rd_u32(const char* p) {
    return static_cast<uint32_t>(static_cast<unsigned char>(p[0])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(p[1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(p[2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(p[3])) << 24);
}

constexpr uint32_t kEocdSig = 0x06054b50;   // end of central directory
constexpr uint32_t kCenSig  = 0x02014b50;   // central directory file header
constexpr uint32_t kLocSig  = 0x04034b50;   // local file header

// Inflate a raw-deflate stream (no zlib/gzip wrapper) of known output size.
std::vector<char> inflate_raw(const char* src, size_t src_len, size_t out_len) {
    std::vector<char> out(out_len);
    if (out_len == 0) return out;

    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    // windowBits = -15 selects raw deflate (matches Python zipfile members).
    if (inflateInit2(&zs, -15) != Z_OK) {
        throw std::runtime_error("npz: inflateInit2 failed");
    }
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(src));
    zs.avail_in = static_cast<uInt>(src_len);
    zs.next_out = reinterpret_cast<Bytef*>(out.data());
    zs.avail_out = static_cast<uInt>(out_len);

    int rc = inflate(&zs, Z_FINISH);
    inflateEnd(&zs);
    if (rc != Z_STREAM_END) {
        throw std::runtime_error("npz: inflate did not complete (corrupt member)");
    }
    return out;
}

// Parsed .npy header.
struct NpyHeader {
    char dtype_kind = '\0';   // 'f' float
    int itemsize = 0;         // 8 or 4
    bool fortran_order = false;
    std::vector<size_t> shape;
    size_t data_offset = 0;   // byte offset of raw data within the .npy blob
};

std::string extract_between(const std::string& s, const std::string& key,
                            char open_ch, char close_ch) {
    size_t k = s.find(key);
    if (k == std::string::npos) return {};
    size_t o = s.find(open_ch, k);
    if (o == std::string::npos) return {};
    size_t c = s.find(close_ch, o);
    if (c == std::string::npos) return {};
    return s.substr(o + 1, c - o - 1);
}

NpyHeader parse_npy_header(const char* blob, size_t len) {
    if (len < 10 || std::memcmp(blob, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("npz: member is not a valid .npy blob");
    }
    const unsigned char major = static_cast<unsigned char>(blob[6]);
    size_t header_len = 0;
    size_t header_start = 0;
    if (major == 1) {
        header_len = rd_u16(blob + 8);
        header_start = 10;
    } else {
        // v2/v3 use a 4-byte header length.
        if (len < 12) throw std::runtime_error("npz: truncated .npy header");
        header_len = rd_u32(blob + 8);
        header_start = 12;
    }
    if (header_start + header_len > len) {
        throw std::runtime_error("npz: .npy header exceeds member size");
    }
    std::string header(blob + header_start, header_len);

    NpyHeader h;
    h.data_offset = header_start + header_len;

    // descr value, e.g. '<f8'. The value is the quoted string AFTER the colon
    // following the "descr" key (extract_between would match the key's own
    // quotes, so parse the value explicitly).
    std::string descr;
    {
        size_t k = header.find("descr");
        if (k == std::string::npos) throw std::runtime_error("npz: .npy header missing descr");
        size_t colon = header.find(':', k);
        size_t q1 = (colon == std::string::npos) ? std::string::npos : header.find('\'', colon);
        size_t q2 = (q1 == std::string::npos) ? std::string::npos : header.find('\'', q1 + 1);
        if (q1 == std::string::npos || q2 == std::string::npos) {
            throw std::runtime_error("npz: cannot parse dtype descr");
        }
        descr = header.substr(q1 + 1, q2 - q1 - 1);
    }
    if (descr.size() < 3) throw std::runtime_error("npz: cannot parse dtype descr");
    // descr is like "<f8" (byteorder, kind, size). Little-endian only.
    char byteorder = descr[0];
    if (byteorder == '>') throw std::runtime_error("npz: big-endian arrays unsupported");
    h.dtype_kind = descr[1];
    h.itemsize = std::stoi(descr.substr(2));
    if (h.dtype_kind != 'f' || (h.itemsize != 8 && h.itemsize != 4)) {
        throw std::runtime_error("npz: unsupported dtype '" + descr + "' (need <f4 or <f8)");
    }

    h.fortran_order = header.find("'fortran_order': True") != std::string::npos ||
                      header.find("'fortran_order':True") != std::string::npos;

    // shape tuple, e.g. (582, 64) or (582,) or ()
    std::string shape_str = extract_between(header, "'shape'", '(', ')');
    // tokenise integers
    std::string cur;
    for (char ch : shape_str) {
        if (ch >= '0' && ch <= '9') {
            cur.push_back(ch);
        } else {
            if (!cur.empty()) { h.shape.push_back(std::stoull(cur)); cur.clear(); }
        }
    }
    if (!cur.empty()) h.shape.push_back(std::stoull(cur));
    return h;
}

// Decode a .npy blob into a column-major (rows, cols) Eigen matrix.
Eigen::MatrixXd decode_npy_matrix(const std::vector<char>& blob) {
    NpyHeader h = parse_npy_header(blob.data(), blob.size());

    size_t rows = 0, cols = 0;
    if (h.shape.empty()) {            // scalar -> 1x1
        rows = 1; cols = 1;
    } else if (h.shape.size() == 1) { // (n,) -> n x 1
        rows = h.shape[0]; cols = 1;
    } else if (h.shape.size() == 2) {
        rows = h.shape[0]; cols = h.shape[1];
    } else {
        throw std::runtime_error("npz: only 1-D and 2-D arrays supported");
    }

    const size_t n = rows * cols;
    const size_t need = n * static_cast<size_t>(h.itemsize);
    if (h.data_offset + need > blob.size()) {
        throw std::runtime_error("npz: .npy data shorter than shape implies");
    }
    const char* data = blob.data() + h.data_offset;

    Eigen::MatrixXd M(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));

    auto value_at = [&](size_t flat) -> double {
        if (h.itemsize == 8) {
            double v;
            std::memcpy(&v, data + flat * 8, 8);
            return v;
        } else {
            float v;
            std::memcpy(&v, data + flat * 4, 4);
            return static_cast<double>(v);
        }
    };

    // File data is a flat buffer in C-order unless fortran_order is set.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t flat = h.fortran_order ? (j * rows + i)   // column-major
                                          : (i * cols + j);  // row-major (C)
            M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value_at(flat);
        }
    }
    return M;
}

} // namespace

NpzArchive::NpzArchive(const std::string& path) : path_(path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("npz: cannot open " + path);
    std::streamsize size = f.tellg();
    if (size <= 0) throw std::runtime_error("npz: empty or unreadable file " + path);
    f.seekg(0, std::ios::beg);
    file_bytes_.resize(static_cast<size_t>(size));
    if (!f.read(file_bytes_.data(), size)) {
        throw std::runtime_error("npz: short read on " + path);
    }

    const char* base = file_bytes_.data();
    const size_t total = file_bytes_.size();

    // Locate the End Of Central Directory record by scanning backwards for its
    // signature (comment field can be up to 65535 bytes after the fixed part).
    if (total < 22) throw std::runtime_error("npz: file too small to be a zip: " + path);
    size_t eocd = std::string::npos;
    size_t scan_start = (total >= 22 + 65535) ? (total - 22 - 65535) : 0;
    for (size_t i = total - 22; ; --i) {
        if (rd_u32(base + i) == kEocdSig) { eocd = i; break; }
        if (i == scan_start) break;
    }
    if (eocd == std::string::npos) {
        throw std::runtime_error("npz: end-of-central-directory not found in " + path);
    }

    const uint16_t n_entries = rd_u16(base + eocd + 10);
    const uint32_t cen_size  = rd_u32(base + eocd + 12);
    const uint32_t cen_off   = rd_u32(base + eocd + 16);
    if (static_cast<size_t>(cen_off) + cen_size > total) {
        throw std::runtime_error("npz: central directory out of range in " + path);
    }

    size_t p = cen_off;
    members_.reserve(n_entries);
    for (uint16_t e = 0; e < n_entries; ++e) {
        if (p + 46 > total || rd_u32(base + p) != kCenSig) {
            throw std::runtime_error("npz: malformed central directory entry in " + path);
        }
        Member m;
        m.method    = rd_u16(base + p + 10);
        m.comp_size = rd_u32(base + p + 20);
        m.uncomp_size = rd_u32(base + p + 24);
        const uint16_t fnlen   = rd_u16(base + p + 28);
        const uint16_t extlen  = rd_u16(base + p + 30);
        const uint16_t cmtlen  = rd_u16(base + p + 32);
        m.local_header_offset  = rd_u32(base + p + 42);
        std::string fname(base + p + 46, fnlen);
        // Strip trailing ".npy" to get the logical array name.
        if (fname.size() >= 4 && fname.compare(fname.size() - 4, 4, ".npy") == 0) {
            m.name = fname.substr(0, fname.size() - 4);
        } else {
            m.name = fname;
        }
        members_.push_back(std::move(m));
        p += 46 + fnlen + extlen + cmtlen;
    }
}

const NpzArchive::Member* NpzArchive::find(const std::string& name) const {
    for (const auto& m : members_) {
        if (m.name == name) return &m;
    }
    return nullptr;
}

bool NpzArchive::has(const std::string& name) const {
    return find(name) != nullptr;
}

std::vector<std::string> NpzArchive::names() const {
    std::vector<std::string> out;
    out.reserve(members_.size());
    for (const auto& m : members_) out.push_back(m.name);
    return out;
}

std::vector<char> NpzArchive::read_member_bytes(const Member& m) const {
    const char* base = file_bytes_.data();
    const size_t total = file_bytes_.size();
    const size_t lh = m.local_header_offset;
    if (lh + 30 > total || rd_u32(base + lh) != kLocSig) {
        throw std::runtime_error("npz: bad local header for member " + m.name);
    }
    // Local header's own name/extra lengths locate the data start. (Central
    // directory's copies can differ from the local ones for the extra field.)
    const uint16_t fnlen  = rd_u16(base + lh + 26);
    const uint16_t extlen = rd_u16(base + lh + 28);
    const size_t data_off = lh + 30 + fnlen + extlen;
    if (data_off + m.comp_size > total) {
        throw std::runtime_error("npz: member data out of range for " + m.name);
    }
    const char* comp = base + data_off;

    if (m.method == 0) {              // stored
        return std::vector<char>(comp, comp + m.comp_size);
    } else if (m.method == 8) {       // deflate
        return inflate_raw(comp, m.comp_size, m.uncomp_size);
    }
    throw std::runtime_error("npz: unsupported compression method for member " + m.name);
}

Eigen::MatrixXd NpzArchive::matrix(const std::string& name) const {
    const Member* m = find(name);
    if (!m) throw std::runtime_error("npz: array '" + name + "' not in " + path_);
    return decode_npy_matrix(read_member_bytes(*m));
}

Eigen::VectorXd NpzArchive::vector(const std::string& name) const {
    Eigen::MatrixXd M = matrix(name);
    if (M.cols() == 1) return M.col(0).eval();
    if (M.rows() == 1) return M.row(0).transpose().eval();
    // Fall back to a column-major flatten for unexpected 2-D shapes.
    return Eigen::Map<Eigen::VectorXd>(M.data(), M.size()).eval();
}

// ---- standalone .npy helpers ----

Eigen::MatrixXd load_npy_matrix(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("npy: cannot open " + path);
    std::streamsize size = f.tellg();
    if (size <= 0) throw std::runtime_error("npy: empty or unreadable file " + path);
    f.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    if (!f.read(bytes.data(), size)) throw std::runtime_error("npy: short read on " + path);
    return decode_npy_matrix(bytes);
}

Eigen::VectorXd load_npy_vector(const std::string& path) {
    Eigen::MatrixXd M = load_npy_matrix(path);
    if (M.cols() == 1) return M.col(0).eval();
    if (M.rows() == 1) return M.row(0).transpose().eval();
    return Eigen::Map<Eigen::VectorXd>(M.data(), M.size()).eval();
}

void save_npy_vector(const std::string& path, const Eigen::VectorXd& v) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("npy: cannot write " + path);

    file.write("\x93NUMPY", 6);
    file.put(0x01);
    file.put(0x00);

    std::ostringstream header;
    header << "{'descr': '<f8', 'fortran_order': False, 'shape': ("
           << v.size() << ",), }";
    std::string header_str = header.str();
    size_t total_header_len = 6 + 2 + 2 + header_str.size();
    size_t padding = (64 - (total_header_len % 64)) % 64;
    header_str.append(padding, ' ');
    header_str.push_back('\n');

    uint16_t header_len = static_cast<uint16_t>(header_str.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header_str.data(), header_len);
    file.write(reinterpret_cast<const char*>(v.data()),
               static_cast<std::streamsize>(v.size() * sizeof(double)));
}

} // namespace npz
