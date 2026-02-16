#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

// Minimal CPU reference for the "legacy stable formulation" used in the CUDA CHR kernel:
// inv_dist = (a^T u) / slack
// alpha ~ Uniform[ 1/min(inv_dist), 1/max(inv_dist) ]
//
// This test constructs a 2D polytope that is thin along the (1,1) direction:
//   |x| <= 1, |y| <= 1, |x+y| <= eps
// Coordinate-only steps have chord length O(eps); pair-direction steps u = (e0 - e1)/sqrt(2)
// are orthogonal to x+y and have O(1) chord length.

static void step_chr_like(
    const std::vector<double>& A_colmajor, // rows x cols, column-major
    const std::vector<double>& b,
    int rows,
    int cols,
    std::vector<double>& x,
    std::vector<double>& slack,
    std::mt19937_64& rng,
    bool use_pair)
{
    std::uniform_real_distribution<double> U(0.0, 1.0);

    int ei = int(U(rng) * cols);
    if (ei >= cols) ei = cols - 1;
    int ej = -1;
    if (use_pair) {
        assert(cols >= 2);
        int r = int(U(rng) * (cols - 1));
        if (r >= cols - 1) r = cols - 2;
        ej = (r >= ei) ? (r + 1) : r;
    }

    double partial_max = -INFINITY;
    double partial_min = INFINITY;
    const double inv_sqrt2 = 0.70710678118654752440;

    for (int r = 0; r < rows; ++r) {
        double s = slack[r];
        double ae;
        if (!use_pair) {
            ae = A_colmajor[ei * rows + r];
        } else {
            const double ai = A_colmajor[ei * rows + r];
            const double aj = A_colmajor[ej * rows + r];
            ae = (ai - aj) * inv_sqrt2;
        }

        double inv_dist;
        if (s == 0.0) {
            if (ae > 0.0) inv_dist = INFINITY;
            else if (ae < 0.0) inv_dist = -INFINITY;
            else inv_dist = 0.0;
        } else {
            inv_dist = ae / s;
            if (std::isnan(inv_dist)) inv_dist = 0.0;
        }
        partial_max = std::max(partial_max, inv_dist);
        partial_min = std::min(partial_min, inv_dist);
    }

    // alpha bounds in the same parameterization as the kernel.
    const double a_lo = 1.0 / partial_min;
    const double a_hi = 1.0 / partial_max;
    const double u = U(rng);
    const double alpha = a_lo + u * (a_hi - a_lo);

    if (!use_pair) {
        x[ei] += alpha;
    } else {
        x[ei] += alpha * inv_sqrt2;
        x[ej] -= alpha * inv_sqrt2;
    }

    // Update slack: slack -= alpha * (a^T u)
    for (int r = 0; r < rows; ++r) {
        double ae;
        if (!use_pair) {
            ae = A_colmajor[ei * rows + r];
        } else {
            const double ai = A_colmajor[ei * rows + r];
            const double aj = A_colmajor[ej * rows + r];
            ae = (ai - aj) * inv_sqrt2;
        }
        slack[r] -= alpha * ae;
    }

    // Basic feasibility sanity: slacks should stay nonnegative for a correct implementation.
    // Allow tiny numerical negatives.
    for (int r = 0; r < rows; ++r) {
        if (slack[r] < -1e-9) {
            std::cerr << "infeasible slack at row " << r << ": " << slack[r] << "\n";
            assert(false);
        }
    }
}

int main() {
    const int rows = 6;
    const int cols = 2;
    const double eps = 1e-4;

    // Constraints:
    //  x <= 1
    // -x <= 1
    //  y <= 1
    // -y <= 1
    //  x+y <= eps
    // -(x+y) <= eps
    //
    // A is rows x cols, column-major.
    // Row order: [x, -x, y, -y, x+y, -(x+y)]
    std::vector<double> A_colmajor(rows * cols, 0.0);
    // column 0 (x)
    A_colmajor[0 * rows + 0] = 1.0;
    A_colmajor[0 * rows + 1] = -1.0;
    A_colmajor[0 * rows + 4] = 1.0;
    A_colmajor[0 * rows + 5] = -1.0;
    // column 1 (y)
    A_colmajor[1 * rows + 2] = 1.0;
    A_colmajor[1 * rows + 3] = -1.0;
    A_colmajor[1 * rows + 4] = 1.0;
    A_colmajor[1 * rows + 5] = -1.0;

    std::vector<double> b = {1.0, 1.0, 1.0, 1.0, eps, eps};

    auto run = [&](bool use_pair) {
        std::vector<double> x = {0.0, 0.0};
        std::vector<double> slack(rows, 0.0);
        for (int r = 0; r < rows; ++r) {
            slack[r] = b[r]; // at x=0, Ax=0 => slack=b
        }
        std::mt19937_64 rng(123);
        double max_abs_diff = 0.0; // max |x-y|
        for (int t = 0; t < 200; ++t) {
            step_chr_like(A_colmajor, b, rows, cols, x, slack, rng, use_pair);
            max_abs_diff = std::max(max_abs_diff, std::abs(x[0] - x[1]));
        }
        return max_abs_diff;
    };

    const double max_abs_diff_coord = run(false);
    const double max_abs_diff_pair = run(true);

    // Coordinate-only moves are limited by the thin slab: |x+y|<=eps implies each coord step is O(eps).
    // Pair-direction (x-y) is orthogonal to the slab and should achieve O(1) movement quickly.
    if (!(max_abs_diff_coord < 1e-2)) {
        std::cerr << "expected coordinate-only chain to remain near x==y; got max|x-y|=" << max_abs_diff_coord << "\n";
        return 1;
    }
    if (!(max_abs_diff_pair > 1e-1)) {
        std::cerr << "expected pair-direction chain to move along x-y; got max|x-y|=" << max_abs_diff_pair << "\n";
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}



