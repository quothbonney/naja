#include "pipeline/extra_constraints.h"
#include "util/start_feasibility.h"

#include <Eigen/Dense>

#include <stdexcept>

int main() {
    // Base: box in 1D: -1 <= x <= 1
    Eigen::MatrixXd A(2, 1);
    A << 1.0,
        -1.0;
    Eigen::VectorXd b(2);
    b << 1.0, 1.0;
    Eigen::VectorXd x0(1);
    x0 << 0.0;

    // Extra constraints that make x0 infeasible: x <= -2
    Eigen::MatrixXd Ae(1, 1);
    Ae << 1.0;
    Eigen::VectorXd be(1);
    be << -2.0;

    // Ignore mode: should not append, x0 remains feasible.
    {
        Eigen::MatrixXd A2 = A;
        Eigen::VectorXd b2 = b;
        bool loaded = naja::pipeline::maybe_augment_extra_constraints(
            A2, b2, &Ae, &be, naja::pipeline::ExtraConstraintsMode::Ignore, 0.0
        );
        if (loaded) throw std::runtime_error("expected ignore mode to not load extra constraints");
        naja::util::require_feasible_start(A2, b2, x0, 1e-9, "ignore");
    }

    // Auto mode: should append and then feasibility check should fail.
    {
        Eigen::MatrixXd A2 = A;
        Eigen::VectorXd b2 = b;
        bool loaded = naja::pipeline::maybe_augment_extra_constraints(
            A2, b2, &Ae, &be, naja::pipeline::ExtraConstraintsMode::Auto, 0.0
        );
        if (!loaded) throw std::runtime_error("expected auto mode to load extra constraints");
        bool ok = false;
        try {
            naja::util::require_feasible_start(A2, b2, x0, 1e-9, "auto");
        } catch (const std::runtime_error&) {
            ok = true;
        }
        if (!ok) throw std::runtime_error("expected feasibility failure under auto mode");
    }

    return 0;
}



