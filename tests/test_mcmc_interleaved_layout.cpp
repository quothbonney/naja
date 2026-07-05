#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

#include "validate/mcmc.h"

int main() {
    const int dim = 1;
    const int n_chains = 2;
    const int spc = 100;
    Eigen::MatrixXf samples(dim, n_chains * spc);

    // Naja CHR writes columns as draw-major, chain-minor:
    // sample t, chain 0; sample t, chain 1; then next t.
    // These two chains have very different means.  If validation incorrectly
    // assumes contiguous chain blocks, it mixes the chains and R-hat looks fine.
    for (int t = 0; t < spc; ++t) {
        const float wiggle = 0.1f * static_cast<float>(std::sin((double)t));
        samples(0, t * n_chains + 0) = wiggle;
        samples(0, t * n_chains + 1) = 10.0f + wiggle;
    }

    auto rhat = naja::validate::compute_split_rhat(samples, n_chains);
    if (!(rhat.rhat[0] > 10.0f)) {
        throw std::runtime_error("split-R-hat did not respect interleaved chain layout");
    }

    auto ess = naja::validate::compute_ess(samples, n_chains);
    if (!(ess.ess[0] > 1.0f)) {
        throw std::runtime_error("ESS did not compute on interleaved chain layout");
    }

    return 0;
}
