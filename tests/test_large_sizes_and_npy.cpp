#ifdef NDEBUG
#undef NDEBUG
#endif
#include "checked_size.h"
#include "engine/profile.h"
#include "npy.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>

int main() {
    using naja::checked_matrix_bytes;
    assert(checked_matrix_bytes<double>(65536, 65536, "test") == UINT64_C(34359738368));
    bool threw = false;
    try { naja::checked_size_mul(std::numeric_limits<std::size_t>::max(), 2, "test"); }
    catch (const std::overflow_error&) { threw = true; }
    assert(threw);
    threw = false;
    try { naja::checked_int_product(65536, 65536, "test"); }
    catch (const std::overflow_error&) { threw = true; }
    assert(threw);
    threw = false;
    try { checked_matrix_bytes<double>(-1, 3, "test"); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    Eigen::MatrixXd samples(2, 3);
    samples << 1.0 + 0x1p-40, -3.0, 4.0, 2.0 - 0x1p-40, 5.0, -6.0;
    npy::save_f64_samples("large_size_test_f64.npy", samples);
    Eigen::MatrixXd restored = npy::load_transposed("large_size_test_f64.npy");
    assert(restored.rows() == samples.rows() && restored.cols() == samples.cols());
    assert(std::memcmp(restored.data(), samples.data(), samples.size() * sizeof(double)) == 0);
    assert(static_cast<double>(static_cast<float>(samples(0, 0))) != samples(0, 0));
    npy::save_f32_samples("large_size_test_f32.npy", samples);
    std::ifstream f32("large_size_test_f32.npy", std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(f32)), std::istreambuf_iterator<char>());
    assert(bytes.find("'<f4'") != std::string::npos);
    assert(bytes.find("(3, 2)") != std::string::npos);
    std::remove("large_size_test_f64.npy");
    std::remove("large_size_test_f32.npy");

    ProfileData profile;
    profile.n_chains = 65536;
    profile.n_samples = 65536;
    profile.total_time = profile.sampling_time = 1.;
    RuntimeConfig config;
    profile.write_json("large_size_test_profile.json", config);
    std::ifstream in("large_size_test_profile.json");
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    assert(text.find("\"total_samples\": 4294967296") != std::string::npos);
    std::remove("large_size_test_profile.json");
    return 0;
}
