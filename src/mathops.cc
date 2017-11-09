#include <cmath>

#include "tensor.h"
#include "mathops.h"

#define CREATE_OP(func_name) void func_name(Tensor& source, Tensor& dest, TensorError* error) { \
  if(!matchedDimensions(source, dest)) { \
    *error = DimensionMismatchError; \
    return; \
  } \
  MultiIndexIterator destIterator(dest.shape, dest.numDimensions); \
  do { \
    uint32_t* currentCoords = destIterator.get(); \
    dest.at(currentCoords) = \
      ::func_name(source.at(currentCoords)); \
  } while(destIterator.next()); \
}

#define CREATE_BINARY_OP(func_name) void func_name(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) { \
  if(!compatibleDimensions(source1, source2)) { \
    *error = DimensionMismatchError; \
    return; \
  } \
  if(!isBroadcastDimension(source1, source2, dest)) {\
    *error = DimensionMismatchError; \
    return; \
  } \
  MultiIndexIterator destIterator(dest.shape, dest.numDimensions); \
  uint32_t numCoords = dest.numDimensions; \
  do { \
    uint32_t* currentCoords = destIterator.get(); \
    dest.at(currentCoords) = \
      ::func_name(source1.broadcast_at(currentCoords, numCoords), source2.broadcast_at(currentCoords, numCoords)); \
  } while(destIterator.next()); \
}

namespace tensor {

void apply(double(*func)(double), Tensor& source, Tensor& dest, TensorError* error) {
  if(!matchedDimensions(source, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  MultiIndexIterator destIterator(dest.shape, dest.numDimensions);
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      (*func)(source.at(currentCoords));
  } while(destIterator.next());
}

CREATE_OP(exp)
CREATE_OP(abs)
CREATE_OP(sqrt)
CREATE_OP(sin)
CREATE_OP(cos)
CREATE_OP(tan)
CREATE_OP(sinh)
CREATE_OP(cosh)
CREATE_OP(tanh)
CREATE_OP(log)
CREATE_OP(atan)
CREATE_OP(acos)
CREATE_OP(asin)
CREATE_OP(atanh)
CREATE_OP(acosh)
CREATE_OP(asinh)
CREATE_OP(erf)
CREATE_OP(floor)
CREATE_OP(ceil)
CREATE_OP(round)

CREATE_BINARY_OP(pow)
CREATE_BINARY_OP(fmod)



} //namespace tensor