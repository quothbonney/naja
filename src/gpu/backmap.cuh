#pragma once

#include "dmatrix.h"
#include "dvector.h"
#include "cublaswrapper.h"
#include "helper.h"

namespace naja::gpu {

template <typename Real>
void backmap_into(const DMatrix<Real> &transformation,
                  const DVector<Real> &shift,
                  const DMatrix<Real> &samples,
                  DMatrix<Real> &output,
                  CUBLASHandle &handle,
                  cudaStream_t stream = 0);

template <typename Real>
DMatrix<Real> backmap(const DMatrix<Real> &transformation,
                      const DVector<Real> &shift,
                      const DMatrix<Real> &samples,
                      CUBLASHandle &handle,
                      cudaStream_t stream = 0);

} // namespace naja::gpu

