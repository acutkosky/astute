#pragma once
#include <cmath>

#include "tensor.h"

#define DECLARE_OP(func_name) void func_name(Tensor& source, Tensor& dest, TensorError* error)


namespace tensor {
  void apply(double(*func)(double), Tensor& source, Tensor& dest, TensorError* error);

  DECLARE_OP(exp);
}