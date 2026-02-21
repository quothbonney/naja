#pragma once

// Declarations to GPU samplers that will be exposed to Python
// Actual implementation in 

#include <functional>

#include "dvector.h"
#include "dmatrix.h"

namespace naja {
    namespace gpu {
        DMatrix<double> WarumUp(DMatrix<double>& A_d,
                                DVector<double>& b_d,
                                DVector<double>& x0_d,
                                int nwarmup,
                                int nchains,
                                int tpb_rd = -1,
                                int tpb_ss = -1);

        DMatrix<double> MatrixHitAndRun     (   DMatrix<double>& A_d,
                                                DVector<double>& b_d,
                                                DMatrix<double>& X_d,
                                                int nspc,
                                                int thinning,
                                                int nchains,
                                                int tpb_rd = -1,
                                                int tpb_ss = -1);

        DMatrix<double> CoordinateHitAndRun(    DMatrix<double>& A_d,
                                                DVector<double>& b_d,
                                                DMatrix<double>& X_d,
                                                int nspc,
                                                int thinning,
                                                int nchains,
                                                int tpb_ss = -1,
                                                int seed = 0,
                                                double pair_prob = 0.0,
                                                int resync_interval = 0,
                                                double ksparse_prob = 0.0,
                                                int ksparse_k = 8,
                                                int pair_mode = 1,
                                                const DVector<int>* pair_i = nullptr,
                                                const DVector<int>* pair_j = nullptr,
                                                const DVector<double>* pair_c = nullptr,
                                                const DVector<double>* pair_s = nullptr,
                                                double dikin_prob = 0.0,
                                                const DMatrix<double>* dikin_Av = nullptr,
                                                const DMatrix<double>* dikin_V = nullptr,
                                                int n_dikin_dirs = 0);

        DMatrix<double> CoordinateHitAndRunBackmap(    DMatrix<double>& A_d,
                                                       DVector<double>& b_d,
                                                       DMatrix<double>& X_d,
                                                       const DMatrix<double>& transformation_d,
                                                       const DVector<double>& shift_d,
                                                       int nspc,
                                                       int thinning,
                                                       int nchains,
                                                       int tpb_ss = -1,
                                                       int seed = 0,
                                                       double pair_prob = 0.0,
                                                       int resync_interval = 0,
                                                       double ksparse_prob = 0.0,
                                                       int ksparse_k = 8,
                                                       int pair_mode = 1,
                                                       const DVector<int>* pair_i = nullptr,
                                                       const DVector<int>* pair_j = nullptr,
                                                       const DVector<double>* pair_c = nullptr,
                                                       const DVector<double>* pair_s = nullptr);

        void CoordinateHitAndRunStreamed(       DMatrix<double>& A_d,
                                                DVector<double>& b_d,
                                                DMatrix<double>& X_d,
                                                int nspc_total,
                                                int chunk_nspc,
                                                int thinning,
                                                int nchains,
                                                int tpb_ss,
                                                int seed,
                                                double pair_prob,
                                                int resync_interval,
                                                double ksparse_prob,
                                                int ksparse_k,
                                                int pair_mode,
                                                const DVector<int>* pair_i,
                                                const DVector<int>* pair_j,
                                                const DVector<double>* pair_c,
                                                const DVector<double>* pair_s,
                                                const std::function<void(const double*, int, int)>& host_sink);

        DVector<double> rhat(const DMatrix<double>& samples_d,
                             int nspc,
                             int nchains,
                             int dim);

        // Minimal CRHMC-style sampler (prototype):
        // - Samples directly in original space using a barrier metric for box bounds
        // - Enforces equalities via projector (I-P) using dense CG solves
        // Inputs:
        //   A_d: equality matrix A (m x n)
        //   l_d, u_d: bounds (length n)
        //   X0_d: initial states (n x nchains), must satisfy A x = b implicitly
        //   nspc: samples per chain
        //   thinning: steps between saved samples
        //   nchains: number of chains (columns of X0_d)
        //   h: integrator step size
        // Output: samples of shape n x (nchains * nspc)
        DMatrix<double> CRHMCPrototype( DMatrix<double>& A_d,
                                         DVector<double>& l_d,
                                         DVector<double>& u_d,
                                         DMatrix<double>& X0_d,
                                         int nspc,
                                         int thinning,
                                         int nchains,
                                         double h);
    }

}
