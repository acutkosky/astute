#pragma once
#include <cmath>

#include "tensor.h"

#define DECLARE_OP(func_name) void func_name(Tensor& source, Tensor& dest, TensorError* error);

#define DECLARE_BINARY_OP(func_name) void func_name(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error);

namespace tensor {

void apply(double(*func)(double), Tensor& source, Tensor& dest, TensorError* error);

DECLARE_OP(exp)
DECLARE_OP(abs)
DECLARE_OP(sqrt)
DECLARE_OP(sin)
DECLARE_OP(cos)
DECLARE_OP(tan)
DECLARE_OP(sinh)
DECLARE_OP(cosh)
DECLARE_OP(tanh)
DECLARE_OP(log)
DECLARE_OP(atan)
DECLARE_OP(acos)
DECLARE_OP(asin)
DECLARE_OP(atanh)
DECLARE_OP(acosh)
DECLARE_OP(asinh)
DECLARE_OP(erf)
DECLARE_OP(floor)
DECLARE_OP(ceil)
DECLARE_OP(round)
DECLARE_OP(sign)

DECLARE_BINARY_OP(max)
DECLARE_BINARY_OP(min)
DECLARE_BINARY_OP(pow)
DECLARE_BINARY_OP(fmod)

}

#undef DECLARE_OP
#undef DECLARE_BINARY_OP