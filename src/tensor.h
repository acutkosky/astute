#pragma once
#include <iostream>
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
  * dimensionsInReversedOrder is a flag that (when true) indicates that
  * dimensions contains dimension sizes in REVERSE order:
  * a MxNxK tensor has 
  * dimensions = [K, N, M]
  *
  * We always maintain the guarantee that the order in dimensions
  * represents the layout of the data in the data array. That is,
  * if dimensions = [K, N, M], then 
  * data[k + nK + mKN] is either the mnk th element of the Tensor if 
  * dimensionsInReversedOrder=true or the
  * knm th element otherwise.
  * this is useful for copy-free transposing.
  */
struct Tensor {
  double* data;
  int numDimensions;
  int* dimensions;
  int* strides;
  int initial_offset;

  int totalSize(void);

  double& at(int* coords, TensorError* error=&globalError);

  // double& at(int* prefixCoords, int* suffixCoords, int suffixSize);

  double& broadcast_at(int* coords, int numCoords, TensorError* error=&globalError);

  void setStrides(bool dimensionsInReversedOrder);
};

struct TensorIterator {
  Tensor* T;
  double* iterator;
  int* currentCoords;
  bool ended;

  TensorIterator(Tensor& t) {
    T = &t;
    currentCoords = new int[T->numDimensions];
    ended = false;
    for(int i=0; i<T->numDimensions; i++) {
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
  int* dimensions;
  int numDimensions;
  int* currentCoords;
  bool ended;

  MultiIndexIterator(int* _dimensions, int _numDimensions) {
    dimensions = _dimensions;
    numDimensions = _numDimensions;
    currentCoords = new int[numDimensions];
    ended = false;
    for(int i=0; i<numDimensions; i++) {
      currentCoords[i] = 0;
    }
  }

  ~MultiIndexIterator() {
    delete [] currentCoords;
  }

  bool next(void);
  int* get(void);
};

void contract(Tensor& source1, Tensor& source2, Tensor& dest, int dimsToContract, TensorError* error=&globalError);

double scalarProduct(Tensor& t1, Tensor& t2, TensorError* error);

void subTensor(Tensor& source, int* heldCoords, int* heldValues, int numHeld, Tensor& dest, TensorError* error=&globalError);

bool matchedDimensions(Tensor& t1, Tensor& t2);

bool compatibleDimensions(Tensor& t1, Tensor& t2);

void transpose(Tensor& source, Tensor& dest, TensorError* error=&globalError);

void addScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale1, double scale2, TensorError* error);

void multiplyScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest, TensorError* error);

void divideScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest, TensorError* error);

void add(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error);

void subtract(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error);

void multiply(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error);

void divide(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error);

void scale(Tensor& source, Tensor& dets, double scale);

void matMul(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error);

} //namespace tensor