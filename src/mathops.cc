#include <cmath>

#include "tensor.h"
#include "mathops.h"

#define CREATE_OP(func_name) void func_name(Tensor& source, Tensor& dest, TensorError* error) { \
  if(!matchedDimensions(source, dest)) { \
    *error = DimensionMismatchError; \
    return; \
  } \
  MultiIndexIterator destIterator(dest.dimensions, dest.numDimensions); \
  uint32_t numDim = dest.numDimensions; \
  do { \
    uint32_t* currentCoords = destIterator.get(); \
    dest.at(currentCoords) = \
      ::func_name(source.at(currentCoords)); \
  } while(destIterator.next()); \
}

namespace tensor {

void apply(double(*func)(double), Tensor& source, Tensor& dest, TensorError* error) {
  if(!matchedDimensions(source, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  MultiIndexIterator destIterator(dest.dimensions, dest.numDimensions);
  uint32_t numDim = dest.numDimensions;
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      (*func)(source.at(currentCoords));
  } while(destIterator.next());
}

void pow(Tensor& source, double exponent, Tensor& dest, TensorError* error) {
  if(!matchedDimensions(source, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  MultiIndexIterator destIterator(dest.dimensions, dest.numDimensions);
  uint32_t numDim = dest.numDimensions;
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      ::pow(source.at(currentCoords), exponent);
  } while(destIterator.next());
}

CREATE_OP(exp)
CREATE_OP(abs)
CREATE_OP(sqrt)
CREATE_OP(sin)
CREATE_OP(cos)
CREATE_OP(sinh)
CREATE_OP(log)



} //namespace