#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <cstddef>
#include "checked_size.h"
#include "error_check.h"

// Widen before either multiplication or addition, on both host and device.
#define IDX2C(i,j,ld) ((static_cast<std::size_t>(j) * static_cast<std::size_t>(ld)) + static_cast<std::size_t>(i))
