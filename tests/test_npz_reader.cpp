// Reads an .npz produced by tools/make_npz_fixture (via the shell wrapper) and
// checks that the C++ npz reader decodes the known arrays correctly.
//
// Fixture contents (see test_npz_reader.sh):
//   A     : float64 (2, 3) = [[1,2,3],[4,5,6]]
//   b     : float64 (2,)   = [7, 8]
//   start : float64 (3,)   = [0.1, 0.2, 0.3]
//   Af    : float32 (2, 3) = same as A  (exercises f4 upcast)

#include "npz.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

bool close(double a, double b) { return std::fabs(a - b) < 1e-9; }

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_npz_reader <fixture.npz>\n";
        return 2;
    }
    const std::string path = argv[1];

    npz::NpzArchive z(path);

    check(z.has("A"), "has A");
    check(z.has("b"), "has b");
    check(z.has("start"), "has start");
    check(!z.has("nope"), "missing array reported absent");

    Eigen::MatrixXd A = z.matrix("A");
    check(A.rows() == 2 && A.cols() == 3, "A shape 2x3");
    check(close(A(0, 0), 1) && close(A(0, 2), 3) && close(A(1, 0), 4) && close(A(1, 2), 6),
          "A values (C-order row-major decode)");

    Eigen::VectorXd b = z.vector("b");
    check(b.size() == 2 && close(b(0), 7) && close(b(1), 8), "b vector");

    Eigen::VectorXd s = z.vector("start");
    check(s.size() == 3 && close(s(0), 0.1) && close(s(2), 0.3), "start vector");

    if (z.has("Af")) {
        Eigen::MatrixXd Af = z.matrix("Af");
        check(Af.rows() == 2 && Af.cols() == 3, "Af shape 2x3");
        check(close(Af(0, 0), 1) && close(Af(1, 2), 6), "Af float32 upcast values");
    }

    if (failures == 0) {
        std::cout << "test_npz_reader OK (" << path << ")\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
