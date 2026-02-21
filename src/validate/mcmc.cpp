#include "validate/mcmc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

extern "C" {
#include "kiss_fftr.h"
}

namespace naja::validate {
namespace {

// FFT-based autocovariance for a single 1D chain.
// Returns acov[0..n-1] where acov[k] = (1/n) sum_{t=0}^{n-1-k} (x_t - mean)(x_{t+k} - mean).
static std::vector<float> autocovariance_fft(const float* data, int n) {
    // Zero-pad to 2n for linear (not circular) correlation
    const int nfft = 2 * n;
    kiss_fftr_cfg fwd = kiss_fftr_alloc(nfft, 0, nullptr, nullptr);
    kiss_fftr_cfg inv = kiss_fftr_alloc(nfft, 1, nullptr, nullptr);

    // Compute mean
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += data[i];
    const float mean = static_cast<float>(sum / n);

    // Center and zero-pad
    std::vector<float> buf(nfft, 0.0f);
    for (int i = 0; i < n; ++i) buf[i] = data[i] - mean;

    // Forward FFT (real -> complex)
    std::vector<kiss_fft_cpx> freq(nfft / 2 + 1);
    kiss_fftr(fwd, buf.data(), freq.data());

    // Power spectrum (|F|^2)
    for (auto& c : freq) {
        float re = c.r, im = c.i;
        c.r = re * re + im * im;
        c.i = 0.0f;
    }

    // Inverse FFT (complex -> real)
    std::vector<float> acov_raw(nfft);
    kiss_fftri(inv, freq.data(), acov_raw.data());

    // Normalize: divide by nfft (FFT convention) and by n (autocovariance convention)
    std::vector<float> acov(n);
    const float norm = 1.0f / (static_cast<float>(nfft) * static_cast<float>(n));
    for (int k = 0; k < n; ++k) {
        acov[k] = acov_raw[k] * norm;
    }

    kiss_fftr_free(fwd);
    kiss_fftr_free(inv);
    return acov;
}

// Geyer's initial positive sequence ESS estimator for one chain.
static float ess_one_chain(const float* data, int n) {
    if (n < 4) return static_cast<float>(n);

    auto acov = autocovariance_fft(data, n);
    if (acov[0] < 1e-30f) return 1.0f;

    // Normalized autocorrelation
    const float var = acov[0];
    float tau = 1.0f;
    for (int k = 1; k < n / 2; ++k) {
        float pair_sum = acov[2 * k - 1] / var + acov[2 * k] / var;
        if (pair_sum < 0.0f) break;
        tau += 2.0f * pair_sum;
    }
    tau = std::max(tau, 1.0f);
    return static_cast<float>(n) / tau;
}

static float percentile(std::vector<float>& v, float q) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    float pos = q * static_cast<float>(v.size() - 1);
    int lo = static_cast<int>(std::floor(pos));
    int hi = static_cast<int>(std::ceil(pos));
    if (lo == hi) return v[lo];
    float t = pos - static_cast<float>(lo);
    return v[lo] * (1.0f - t) + v[hi] * t;
}

} // namespace

EssResult compute_ess(const Eigen::MatrixXf& samples, int n_chains) {
    const int dim = samples.rows();
    const int n_total = samples.cols();
    const int spc = n_total / n_chains;
    if (spc < 4) throw std::invalid_argument("compute_ess: fewer than 4 samples per chain");

    EssResult result;
    result.ess.resize(dim);

    for (int d = 0; d < dim; ++d) {
        float min_ess = std::numeric_limits<float>::max();
        for (int c = 0; c < n_chains; ++c) {
            const float* chain_data = samples.data() + static_cast<long>(d) + static_cast<long>(c * spc) * dim;
            // Eigen col-major: column c*spc+i has element d at offset d + (c*spc+i)*dim
            // We need contiguous chain data, so copy it out
            std::vector<float> chain(spc);
            for (int i = 0; i < spc; ++i) {
                chain[i] = samples(d, c * spc + i);
            }
            float e = ess_one_chain(chain.data(), spc);
            min_ess = std::min(min_ess, e);
        }
        result.ess[d] = min_ess;
    }

    // Compute summary stats
    std::vector<float> ess_vec(result.ess.data(), result.ess.data() + dim);
    result.min = *std::min_element(ess_vec.begin(), ess_vec.end());
    result.max = *std::max_element(ess_vec.begin(), ess_vec.end());
    result.p10 = percentile(ess_vec, 0.10f);
    result.median = percentile(ess_vec, 0.50f);

    return result;
}

RhatResult compute_split_rhat(const Eigen::MatrixXf& samples, int n_chains) {
    const int dim = samples.rows();
    const int n_total = samples.cols();
    const int spc = n_total / n_chains;
    const int half = spc / 2;
    if (half < 2) throw std::invalid_argument("compute_split_rhat: chain too short to split");

    const int m = 2 * n_chains;  // number of half-chains

    RhatResult result;
    result.rhat.resize(dim);
    result.above_1_1 = 0;
    result.above_1_2 = 0;

    for (int d = 0; d < dim; ++d) {
        // Compute per-half-chain mean and variance
        double grand_sum = 0.0;
        std::vector<double> chain_means(m);
        std::vector<double> chain_vars(m);

        for (int c = 0; c < n_chains; ++c) {
            for (int h = 0; h < 2; ++h) {
                int start = c * spc + h * half;
                double s = 0.0, s2 = 0.0;
                for (int i = 0; i < half; ++i) {
                    double v = static_cast<double>(samples(d, start + i));
                    s += v;
                    s2 += v * v;
                }
                double mean = s / half;
                double var = (s2 / half) - mean * mean;
                // Bessel correction
                var = var * half / (half - 1);
                chain_means[2 * c + h] = mean;
                chain_vars[2 * c + h] = var;
                grand_sum += mean;
            }
        }

        double grand_mean = grand_sum / m;

        // Between-chain variance B
        double B = 0.0;
        for (int j = 0; j < m; ++j) {
            double diff = chain_means[j] - grand_mean;
            B += diff * diff;
        }
        B *= static_cast<double>(half) / (m - 1);

        // Within-chain variance W
        double W = 0.0;
        for (int j = 0; j < m; ++j) W += chain_vars[j];
        W /= m;

        // Var estimate and R-hat
        double var_hat = ((half - 1.0) / half) * W + B / half;
        double rhat = (W > 1e-30) ? std::sqrt(var_hat / W) : 1.0;
        result.rhat[d] = static_cast<float>(rhat);

        if (rhat > 1.1) ++result.above_1_1;
        if (rhat > 1.2) ++result.above_1_2;
    }

    std::vector<float> rhat_vec(result.rhat.data(), result.rhat.data() + dim);
    result.median = percentile(rhat_vec, 0.50f);
    result.max = *std::max_element(rhat_vec.begin(), rhat_vec.end());

    return result;
}

} // namespace naja::validate

