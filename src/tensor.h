#pragma once
#include <iostream>
#include <stdint.h>
using std::cout;
using std::endl;

namespace tensor {

#define ASSERT(expr, error) if(!(expr)) { return error; }
#define ASSERT_SIZE_EQUAL_3(a, b, c)  ASSERT(a.totalSize() == b.totalSize() && a.totalSize() == c.totalSize(), SizeMismatchError)
#define MIN(a,b) (a>b?b:a)
#define MAX(a,b) (a>b?a:b)


enum TensorError {
  NoError = 0,
  SizeMismatchError,
  DimensionMismatchError,
  IndexOutOfBounds,
  MemoryLeakError
};


extern TensorError globalError;

/**
  * shapeInReversedOrder is a flag that (when true) indicates that
  * shape contains dimension sizes in REVERSE order:
  * a MxNxK tensor has 
  * shape = [K, N, M]
  *
  * We always mauint32_tain the guarantee that the order in shape
  * represents the layout of the data in the data array. That is,
  * if shape = [K, N, M], then 
  * data[k + nK + mKN] is either the mnk th element of the Tensor if 
  * shapeInReversedOrder=true or the
  * knm th element otherwise.
  * this is useful for copy-free transposing.
  */
struct Tensor {
  double* data;
  uint32_t numDimensions;
  uint32_t* shape;
  uint32_t* strides;
  uint32_t initial_offset;

  uint32_t totalSize(void);

  uint32_t maximumOffset(void);

  double& at(uint32_t* coords, TensorError* error=&globalError);

  // double& at(uint32_t* prefixCoords, uint32_t* suffixCoords, uint32_t suffixSize);

  double& broadcast_at(uint32_t* coords, uint32_t numCoords, TensorError* error=&globalError);

  void setStrides(bool shapeInReversedOrder);

  bool isValid(void);
};

struct TensorIterator {
  Tensor* T;
  double* iterator;
  uint32_t* currentCoords;
  bool ended;

  TensorIterator(Tensor& t) {
    T = &t;
    currentCoords = new uint32_t[T->numDimensions];
    ended = false;
    for(uint32_t i=0; i<T->numDimensions; i++) {
      currentCoords[i] = 0;
    }
    iterator = &T->at(currentCoords);
  }

  ~TensorIterator() {
    delete [] currentCoords;
  }

  bool next(void);

  double& get(void);
};

struct MultiIndexIterator {
  uint32_t* shape;
  uint32_t numDimensions;
  uint32_t* currentCoords;
  bool ended;

  MultiIndexIterator(uint32_t* _shape, uint32_t _numDimensions) {
    shape = _shape;
    numDimensions = _numDimensions;
    currentCoords = new uint32_t[numDimensions];
    ended = false;
    for(uint32_t i=0; i<numDimensions; i++) {
      currentCoords[i] = 0;
    }
  }

  ~MultiIndexIterator() {
    delete [] currentCoords;
  }

  bool next(void);
  uint32_t* get(void);
};

void contract(Tensor& source1, Tensor& source2, Tensor& dest, uint32_t dimsToContract, TensorError* error=&globalError);

double scalarProduct(Tensor& t1, Tensor& t2, TensorError* error=&globalError);

void subTensor(Tensor& source, uint32_t* heldCoords, uint32_t* heldValues, uint32_t numHeld, Tensor& dest, TensorError* error=&globalError);

bool matchedDimensions(Tensor& t1, Tensor& t2);

bool compatibleDimensions(Tensor& t1, Tensor& t2);

bool isBroadcastDimension(Tensor& source1, Tensor& source2, Tensor& dest);

void transpose(Tensor& source, Tensor& dest, TensorError* error=&globalError);

void addScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale1, double scale2, TensorError* error=&globalError);

void multiplyScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale, TensorError* error=&globalError);

void divideScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale, TensorError* error=&globalError);

void denseAddScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale1, double scale2);

void denseMultiplyScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale);

void denseDivideScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale);

void add(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error=&globalError);

void subtract(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error=&globalError);

void multiply(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error=&globalError);

void divide(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error=&globalError);

void scale(Tensor& source, Tensor& dest, double scale, TensorError* error=&globalError);

void matMul(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error=&globalError);

void simpleMatMul(Tensor& source1, Tensor& source2, Tensor& dest);

void fastMatMul(Tensor& source1, Tensor& source2, Tensor& dest);

void fastMatVectMul(bool transpose, Tensor& matrix, Tensor& vector, Tensor& dest);

bool isDense(Tensor& source);

} //namespace tensor