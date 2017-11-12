#include <iostream>
#include <chrono>
#include <random>
#include "cblas.h"

#include "tensor.h"
namespace tensor {

TensorError globalError;

using std::cout;
using std::endl;

std::mt19937 global_generator;

void seed_generator(void) {
  try {
    std::random_device rd;
    global_generator.seed(rd());
  } catch(int e) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + 3;
    global_generator.seed(seed);
  }
}

uint32_t Tensor::totalSize(void) {
  if(this->numDimensions == 0) {
    return 0;
  }
  uint32_t accumulator = 1;
  for(uint32_t i=0; i<this->numDimensions; i++) {
    accumulator *= this->shape[i];
  }
  return accumulator;
}

uint32_t Tensor::maximumOffset(void) {
  if(this->numDimensions == 0) {
    return this->initial_offset;
  }
  uint32_t offset = this->initial_offset;

  for(uint32_t i=0; i<this->numDimensions; i++) {
    offset += (this->shape[i]-1)*this->strides[i];
  }

  return offset;
}

bool Tensor::isValid(void) {
  return this->data != NULL;
}

double& Tensor::at(uint32_t* coords, TensorError* error) {
  uint32_t offset = this->initial_offset;
  for(uint32_t i=0; i<this->numDimensions; i++) {
    if(coords[i] >= this->shape[i]) {
      *error = IndexOutOfBounds;
      return this->data[offset];
    }
    offset += coords[i] * this->strides[i];
  }
  return this->data[offset];
}

double& Tensor::broadcast_at(uint32_t* coords, uint32_t numCoords, TensorError* error) {
  uint32_t offset = this->initial_offset;
  for(uint32_t i=0; i<this->numDimensions; i++) {
    uint32_t dimension = this->shape[this->numDimensions - i - 1];
    uint32_t stride = this->strides[this->numDimensions - i - 1];
    uint32_t coord = coords[numCoords -i -1];
    if(coord < dimension) {
      offset += coord * stride;
    } else if(dimension != 1) {
      *error = IndexOutOfBounds;
      return this->data[offset];
    }
  }
  return this->data[offset];
}

/**
  * sets the strides pouint32_ter in the tensor object.
  * shapeInReversedOrder specifies whether the tensor is 
  * stored in "column major" or "row major" order. Specifically, if
  * shapeInReversedOrder is true, the ijk th element of a NxMxK tensor
  * is located at
  *   data[i + j*N + k*MN]
  * and so strides will be set to 
  *   [1, N, MN]
  * if shapeInReversedOrder is true, the ikj element is at
  *   data[i*MK + j*K + k]
  * and so strides will be set to
  * [MK, K, 1]
  *
  * It is not the business of this library to manage memory that might
  * be used externally; we assume that the strides array already exists.
  **/

void Tensor::setStrides(bool shapeInReversedOrder) {
  if(shapeInReversedOrder) {
    uint32_t currentStride = 1;
    for(uint32_t i=0; i<numDimensions; i++) {
      this->strides[i] = currentStride;
      currentStride *= this->shape[i];
    }
  } else {
    uint32_t currentStride = 1;
    for(int i=this->numDimensions-1; i>=0; i--) {
      this->strides[i] = currentStride;
      currentStride *= this->shape[i];
    }
  }
}

void transpose(Tensor& source, Tensor& dest, TensorError* error) {
  if(source.data != dest.data) {
    *error = MemoryLeakError;
    return;
  }
  if(source.numDimensions != dest.numDimensions) {
    *error = SizeMismatchError;
    return;
  }
  for(uint32_t i=0; i<source.numDimensions; i++) {
    dest.shape[i] = source.shape[source.numDimensions-1-i];
    dest.strides[i] = source.strides[source.numDimensions-1-i];
  }
  dest.initial_offset = source.initial_offset;
}


bool matchedDimensions(Tensor& t1, Tensor& t2) {
  if(t1.numDimensions != t2.numDimensions)
    return false;
  if(t1.numDimensions == 0)
    return true;

  for(uint32_t i=0; i<t1.numDimensions; i++) {
    if(t1.shape[i] != t2.shape[i])
      return false;
  }

  return true;
}

bool compatibleDimensions(Tensor& t1, Tensor& t2) {
  uint32_t minimumDim = MIN(t1.numDimensions, t2.numDimensions);

  for(uint32_t i=0; i<minimumDim; i++) {
    if(t1.shape[t1.numDimensions-1-i] == 1)
      continue;
    if(t2.shape[t2.numDimensions-1-i] == 1)
      continue;
    if(t1.shape[t1.numDimensions-1-i] == t2.shape[t2.numDimensions-1-i])
      continue;

    return false;
  }

  return true;
}


/**
  * fills in the already-allocated memory in dest to reflect a subtensor
  * of source, using the same data storage.
  * To specify a subTensor we choose a subset of axes of the original
  * tensor to hold constant and let the others vary.
  * Example:
  *   source is a NxMxK tensor
  *   heldCoords = [0,2]
  *   heldValues = [3,4]
  *   numHeld = 2
  *     then dest will be a M tensor (i.e. a M-dimensional vector) with values
  *     dest[m] = source[3, m, 4]
  *
  * We assume heldCoords is sorted.
  **/
void subTensor(Tensor& source, uint32_t* heldCoords, uint32_t* heldValues, uint32_t numHeld, Tensor& dest, TensorError* error) {
  if(dest.data!=source.data) {
    *error = MemoryLeakError;
    return;   
  }
  if(dest.numDimensions != source.numDimensions - numHeld) {
    *error = DimensionMismatchError;
    return;
  }
  uint32_t heldCoordsIndex = 0;
  uint32_t destDimensionIndex = 0;
  uint32_t offset = source.initial_offset;
  for(uint32_t i=0; i<source.numDimensions; i++) {
    if(heldCoordsIndex>= numHeld || i!=heldCoords[heldCoordsIndex]) {
      dest.shape[destDimensionIndex] = source.shape[i];
      dest.strides[destDimensionIndex] = source.strides[i];
      destDimensionIndex++;
    } else {
      offset += source.strides[i] * heldValues[heldCoordsIndex];
      heldCoordsIndex++;
    }
  }
  dest.initial_offset = offset;
}

bool TensorIterator::next(void) {

  if(T->numDimensions == 0) {
    ended = true;
  } else {
   
    uint32_t i = 0;
    int offset = 0;
    currentCoords[i] = (currentCoords[i] + 1) % T->shape[i];
    offset += T->strides[i];
    while(currentCoords[i] == 0) {
      offset -= T->strides[i] * T->shape[i];
      i++;
      if(i>=T->numDimensions) {
        ended = true;
        offset = 0;
        break;
      }

      currentCoords[i] = (currentCoords[i] + 1) % T->shape[i];
      offset += T->strides[i];

     }
     iterator += offset;
  }
  return !ended;
}

double& TensorIterator::get(void) {
  return *iterator;
}


bool MultiIndexIterator::next(void) {
  uint32_t i = 0;
  currentCoords[i] = (currentCoords[i] + 1) % shape[i];
  while(currentCoords[i] == 0) {
    i++;
    if(i>=numDimensions) {
      ended = true;
      break;
    }
    currentCoords[i] = (currentCoords[i] + 1) % shape[i];
   }
  return !ended;
}

uint32_t* MultiIndexIterator::get(void) {
  return currentCoords;
}

double scalarProduct(Tensor& t1, Tensor& t2, TensorError* error) {

  if(!matchedDimensions(t1, t2)) {
    *error = DimensionMismatchError;
    return 0.0;
  }

  TensorIterator iter1(t1);
  TensorIterator iter2(t2);
  double product = 0.0;
  do {
    product += iter1.get() * iter2.get();
  } while(iter1.next() && iter2.next());
  return product;

}

void print2DCoord(uint32_t* coords) {
  cout<<coords[0]<<" "<<coords[1]<<endl;
}

bool compatibleForContraction(Tensor& source1, Tensor& source2, uint32_t dimsToContract, Tensor& dest) {

  if(source1.numDimensions < dimsToContract)
    return false;
  if(source2.numDimensions < dimsToContract)
    return false;


  for(uint32_t i=0; i<dimsToContract; i++) {
    uint32_t source1Offset = source1.numDimensions - 1 - i;
    uint32_t source2Offset = i;
    if(source1.shape[source1Offset] != source2.shape[source2Offset]) 
      return false;
  }

  for(uint32_t i=0; i<source1.numDimensions - dimsToContract; i++) {
    if(dest.shape[i] != source1.shape[i])
      return false;
  }
  for(uint32_t i=0; i<source2.numDimensions - dimsToContract; i++) {
    if(dest.shape[i + source1.numDimensions - dimsToContract] !=
        source2.shape[i + dimsToContract])
      return false;
  }

  return true;
}

void outerProduct(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {

  MultiIndexIterator destIterator(dest.shape, dest.numDimensions);

  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords, error) = source1.at(currentCoords, error) * source2.at(currentCoords + source1.numDimensions, error);

    if(*error != NoError)
      return;

  } while(destIterator.next());
}

void contract(Tensor& source1, Tensor& source2, uint32_t dimsToContract, Tensor& dest, TensorError* error) {

  //Verify dimensions
  if(dest.numDimensions != 
      source1.numDimensions + source2.numDimensions - 2 * dimsToContract) {
    if(!(dest.numDimensions == 1 && source1.numDimensions + source2.numDimensions - 2 * dimsToContract == 0 )) {
      *error = DimensionMismatchError;
      return;
    }
  }

  if(!compatibleForContraction(source1, source2, dimsToContract, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  //check if outer product
  if(dimsToContract == 0) {
    outerProduct(source1, source2, dest, error);
    return;
  }

  //check for special-case speedups using BLAS routines:

  //MM matrix-matrix multiply
  if(source1.numDimensions==2 && source2.numDimensions==2 && dimsToContract==1) {
    fastMatMul(source1, source2, dest);
    return;
  }

  //Mv matrix-vector multiply
  if(source1.numDimensions==2 && source2.numDimensions==1 && dimsToContract==1) {
    fastMatVectMul(false, source1, source2, dest);
    return;
  }

  //vM vector-matrix multiply
  if(source1.numDimensions==1 && source2.numDimensions==2 && dimsToContract==1) {
    fastMatVectMul(true, source2, source1, dest);
    return;
  }

  //vv dot product
  if(source1.numDimensions==1 && source2.numDimensions==1 & dimsToContract==1) {
    fastDotProduct(source1, source2, dest);
    return;
  }


  MultiIndexIterator destIterator(dest.shape, dest.numDimensions);

  uint32_t* dimRange = new uint32_t[dest.numDimensions];
  for(uint32_t i=0; i<dest.numDimensions; i++) {
    dimRange[i] = i;
  }
  Tensor sub1, sub2;

  if(dimsToContract != 0) {
    sub1.shape = new uint32_t[dimsToContract];
    sub1.strides = new uint32_t[dimsToContract];
  }
  sub1.numDimensions = dimsToContract;
  sub1.data = source1.data;

  if(dimsToContract != 0) {
    sub2.shape = new uint32_t[dimsToContract];
    sub2.strides = new uint32_t[dimsToContract];
  }
  sub2.numDimensions = dimsToContract;
  sub2.data = source2.data;

  do {
    uint32_t* currentCoords = destIterator.get();
    subTensor(source1, 
              dimRange, 
              currentCoords, 
              source1.numDimensions - dimsToContract,
              sub1,
              error);
    transpose(sub1, sub1, error);
    subTensor(source2, 
              dimRange + source1.numDimensions - dimsToContract,
              currentCoords + source1.numDimensions - dimsToContract,
              source2.numDimensions - dimsToContract,
              sub2,
              error);

    if(*error != NoError)
      return;

    dest.at(currentCoords, error) = scalarProduct(sub1, sub2, error);

    if(*error != NoError)
      return;

  } while(destIterator.next());

  if(dimsToContract != 0) {
    delete [] sub1.shape;
    delete [] sub1.strides;
    delete [] sub2.shape;
    delete [] sub2.strides;
  }
}

bool isBroadcastDimension(Tensor& source1, Tensor& source2, Tensor& dest) {
  uint32_t maxDimensions = MAX(source1.numDimensions, source2.numDimensions);
  if(dest.numDimensions != maxDimensions)
    return false;

  uint32_t dimension1, dimension2;

  for(uint32_t i=0; i<maxDimensions; i++) {
    dimension1 = dimension2 = 1;
    if(i<source1.numDimensions)
      dimension1 = source1.shape[source1.numDimensions - i -1];

    if(i<source2.numDimensions)
      dimension2= source2.shape[source2.numDimensions - i -1];

    if(dest.shape[maxDimensions - i -1] != MAX(dimension1, dimension2))
      return false;
  }
  return true;
}

bool identicalLayout(Tensor& tensor1, Tensor& tensor2) {
  if(tensor1.numDimensions != tensor2.numDimensions)
    return false;

  for(uint32_t i=0; i<tensor1.numDimensions; i++) {
    if(tensor1.strides[i] != tensor2.strides[i])
      return false;
    if(tensor1.shape[i] != tensor2.shape[i])
      return false;
  }
  return true;
}

void addScale(Tensor& source1, Tensor& source2, double scale1, double scale2, Tensor& dest, TensorError* error) {
  if(!compatibleDimensions(source1, source2)) {
    *error = DimensionMismatchError;
    return;
  }
  if(!isBroadcastDimension(source1, source2, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  if(identicalLayout(dest, source1) && identicalLayout(dest, source2) && isDense(dest)) {
    denseAddScale(source1, source2, scale1, scale2, dest);
    return;
  }

  MultiIndexIterator destIterator(dest.shape, dest.numDimensions);
  uint32_t numDim = dest.numDimensions;
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale1 * source1.broadcast_at(currentCoords, numDim) +
      scale2 * source2.broadcast_at(currentCoords, numDim);
  } while(destIterator.next());

}

void denseAddScale(Tensor& source1, Tensor& source2, double scale1, double scale2, Tensor& dest) {
  uint32_t totalSize = source1.totalSize();
  double* source1Iterator = source1.data + source1.initial_offset;
  double* source2Iterator = source2.data + source2.initial_offset;
  double* destIterator = dest.data + dest.initial_offset;
  for(uint32_t i = 0; i<totalSize; i++) {
    *destIterator = scale1 * (*source1Iterator) + scale2* (*source2Iterator);
    destIterator++;
    source1Iterator++;
    source2Iterator++;
  }
}

void multiplyScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest, TensorError* error) {
  if(!compatibleDimensions(source1, source2)) {
    *error = DimensionMismatchError;
    return;
  }
  if(!isBroadcastDimension(source1, source2, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  if(identicalLayout(dest, source1) && identicalLayout(dest, source2) && isDense(dest)) {
    denseMultiplyScale(source1, source2, scale, dest);
    return;
  }

  MultiIndexIterator destIterator(dest.shape, dest.numDimensions);
  uint32_t numDim = dest.numDimensions;
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale * 
      source1.broadcast_at(currentCoords, numDim) *
      source2.broadcast_at(currentCoords, numDim);
  } while(destIterator.next());

}

void denseMultiplyScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest) {
  uint32_t totalSize = source1.totalSize();
  double* source1Iterator = source1.data + source1.initial_offset;
  double* source2Iterator = source2.data + source2.initial_offset;
  double* destIterator = dest.data + dest.initial_offset;
  for(uint32_t i = 0; i<totalSize; i++) {
    *destIterator = scale * (*source1Iterator) * (*source2Iterator);
    destIterator++;
    source1Iterator++;
    source2Iterator++;
  }
}

void divideScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest, TensorError* error) {
  if(!compatibleDimensions(source1, source2)) {
    *error = DimensionMismatchError;
    return;
  }
  if(!isBroadcastDimension(source1, source2, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  if(identicalLayout(dest, source1) && identicalLayout(dest, source2) && isDense(dest)) {
    denseDivideScale(source1, source2, scale, dest);
    return;
  }

  MultiIndexIterator destIterator(dest.shape, dest.numDimensions);
  uint32_t numDim = dest.numDimensions;
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale * 
      source1.broadcast_at(currentCoords, numDim) /
      source2.broadcast_at(currentCoords, numDim);
  } while(destIterator.next());

}

void denseDivideScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest) {
  uint32_t totalSize = source1.totalSize();
  double* source1Iterator = source1.data + source1.initial_offset;
  double* source2Iterator = source2.data + source2.initial_offset;
  double* destIterator = dest.data + dest.initial_offset;
  for(uint32_t i = 0; i<totalSize; i++) {
    *destIterator = scale * (*source1Iterator) / (*source2Iterator);
    destIterator++;
    source1Iterator++;
    source2Iterator++;
  }
}

void scale(Tensor& source, double scale, Tensor& dest, TensorError* error) {
  if(!matchedDimensions(source, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  if(identicalLayout(dest, source) && isDense(dest)) {
    denseScale(source, scale, dest);
    return;
  }

  MultiIndexIterator destIterator(dest.shape, dest.numDimensions);
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale * source.at(currentCoords);
  } while(destIterator.next());
}

void denseScale(Tensor& source, double scale, Tensor& dest) {
  uint32_t totalSize = source.totalSize();
  double* sourceIterator = source.data + source.initial_offset;
  double* destIterator = dest.data + dest.initial_offset;
  for(uint32_t i = 0; i<totalSize; i++) {
    *destIterator = scale * (*sourceIterator);
    destIterator++;
    sourceIterator++;
  }
}

void add(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return addScale(source1, source2, 1, 1, dest, error);
}

void subtract(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return addScale(source1, source2, 1, -1, dest, error);
}

void multiply(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return multiplyScale(source1, source2, 1, dest, error);
}

void divide(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return divideScale(source1, source2, 1, dest, error);
}

bool isDense(Tensor& source) {
  if(source.strides[0]==1) {
    uint32_t denseStride = 1;
    for(uint32_t i=0; i<source.numDimensions; i++) {
      if(source.strides[i] != denseStride)
        return false;
      denseStride *= source.shape[i];
    }
  } else if(source.strides[source.numDimensions-1] == 1) {
    uint32_t denseStride = 1;
    for(uint32_t i=0; i<source.numDimensions; i++) {
      if(source.strides[source.numDimensions - i -1] != denseStride)
        return false;
      denseStride *= source.shape[source.numDimensions - i - 1];
    }
  } else {
    return false;
  }
  return true;
}

void simpleMatMul(Tensor& source1, Tensor& source2, Tensor& dest) {
  double* destData = dest.data + dest.initial_offset;
  double* source1Data = source1.data + source1.initial_offset;
  double* source2Data = source2.data + source2.initial_offset;

  uint32_t destStrides0 = dest.strides[0];
  uint32_t destStrides1 = dest.strides[1];

  uint32_t source1Strides0 = source1.strides[0];
  uint32_t source1Strides1 = source1.strides[1];

  uint32_t source2Strides0 = source2.strides[0];
  uint32_t source2Strides1 = source2.strides[1];

  uint32_t kMax = source1.shape[1];


  for(uint32_t i=0; i<dest.shape[0]; i++) {
    for(uint32_t j=0; j<dest.shape[1]; j++) {
      double* source1Current = source1Data + i*source1Strides0;
      double* source2Current = source2Data + j*source2Strides1;
      double* destCurrent = destData + i*destStrides0 + j*destStrides1;
      *destCurrent = 0;
      for(uint32_t k=0; k<kMax; k++) {
        *destCurrent += (*source1Current) * (*source2Current);
        source1Current += source1Strides1;
        source2Current += source2Strides0;
      }
    }
  }
}

void fastMatMul(Tensor& source1, Tensor& source2, Tensor& dest) {

  if(!isDense(source1) || !isDense(source2) || !isDense(dest)) {
    simpleMatMul(source1, source2, dest);
    return;
  }

  CBLAS_ORDER order=CblasRowMajor;
  CBLAS_TRANSPOSE transpose1=CblasNoTrans;
  CBLAS_TRANSPOSE transpose2=CblasNoTrans;
  uint32_t source1Stride = MAX(source1.strides[0], source1.strides[1]);
  uint32_t source2Stride = MAX(source2.strides[0], source2.strides[1]);
  uint32_t destStride = MAX(dest.strides[0], dest.strides[1]);
  if(dest.strides[0] == 1) {
    order = CblasColMajor;
    if(source1.strides[0] == 1) {
      transpose1 = CblasNoTrans;
    } else {
      transpose1 = CblasTrans;
    }

    if(source2.strides[0] == 1) {
      transpose2 = CblasNoTrans;
    } else {
      transpose2 = CblasTrans;
    }

  } else if(dest.strides[1] == 1) {
    order = CblasRowMajor;
    if(source1.strides[1] == 1) {
      transpose1 = CblasNoTrans;
    } else {
      transpose1 = CblasTrans;
    }

    if(source2.strides[1] == 1) {
      transpose2 = CblasNoTrans;
    } else {
      transpose2 = CblasTrans;
    }
  }
  cblas_dgemm(order, transpose1, transpose2, source1.shape[0], source2.shape[1], source1.shape[1], 1.0, source1.data+source1.initial_offset, source1Stride, source2.data+source2.initial_offset, source2Stride, 0.0, dest.data+dest.initial_offset, destStride);
}

void simpleMatVectMul(bool transpose, Tensor& matrix, Tensor& vector, Tensor& dest) {
  double* destData = dest.data + dest.initial_offset;
  double* matrixData = matrix.data + matrix.initial_offset;
  double* vectorData = vector.data + vector.initial_offset;

  uint32_t destStride = dest.strides[0];

  uint32_t matrixStrides0 = transpose?matrix.strides[1]:matrix.strides[0];
  uint32_t matrixStrides1 = transpose?matrix.strides[0]:matrix.strides[1];

  uint32_t vectorStride = vector.strides[0];

  uint32_t innerDim = transpose?matrix.shape[0]:matrix.shape[1];


  for(uint32_t i=0; i<dest.shape[0]; i++) {
    double* matrixCurrent = matrixData + i*matrixStrides0;
    double* vectorCurrent = vectorData;
    double* destCurrent = destData + i*destStride;
    *destCurrent = 0;
    for(uint32_t j=0; j<innerDim; j++) {
      *destCurrent += (*matrixCurrent) * (*vectorCurrent);
      matrixCurrent += matrixStrides1;
      vectorCurrent += vectorStride;
    }
  }
}


void fastMatVectMul(bool transpose, Tensor& matrix, Tensor& vector, Tensor& dest) {
  //assume source1 is the matrix and source2 is the tensor.
  //transpose source1 if necessary.
  CBLAS_ORDER order = CblasRowMajor;
  CBLAS_TRANSPOSE cblas_trans = transpose?CblasTrans:CblasNoTrans;
  int matrixStride = MAX(matrix.strides[0], matrix.strides[1]);
  int vectorStride = vector.strides[0];
  int destStride = dest.strides[0];

  if(matrix.strides[0] == 1) {
    order = CblasColMajor;
  } else if(matrix.strides[1] == 1) {
    order = CblasRowMajor;
  } else {
    simpleMatVectMul(transpose, matrix, vector, dest);
    return;
  }
  cblas_dgemv(order, cblas_trans, matrix.shape[0], matrix.shape[1], 1.0, matrix.data+matrix.initial_offset, matrixStride, vector.data+vector.initial_offset, vectorStride, 0, dest.data+dest.initial_offset, destStride);
  return;
}

void fastDotProduct(Tensor& vector1, Tensor& vector2, Tensor& dest) {
  dest.data[0] = cblas_ddot(vector1.shape[0], 
                            vector1.data + vector1.initial_offset,
                            vector1.strides[0],
                            vector2.data + vector2.initial_offset,
                            vector2.strides[0]);
}


void matMul(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return contract(source1, source2, 1, dest, error);
}

void fillNormal(double mean, double std_dev, Tensor& dest) {
  std::normal_distribution<double> distribution(mean, std_dev);
  if(isDense(dest)) {
    uint32_t totalSize = dest.totalSize();
    double* iterator = dest.data + dest.initial_offset;
    for(uint32_t i=0; i<totalSize; i++) {
      iterator[i] = distribution(global_generator);
    }
  } else {
    TensorIterator iterator(dest);
    do {
      iterator.get() = distribution(global_generator);
    } while(iterator.next());
  }
}

void fillUniform(double low, double high, Tensor& dest) {
  std::uniform_real_distribution<double> distribution(low, high);
  if(isDense(dest)) {
    uint32_t totalSize = dest.totalSize();
    double* iterator = dest.data + dest.initial_offset;
    for(uint32_t i=0; i<totalSize; i++) {
      iterator[i] = distribution(global_generator);
    }
  } else {
    TensorIterator iterator(dest);
    do {
      iterator.get() = distribution(global_generator);
    } while(iterator.next());
  }
}

double sum(Tensor& source) {
  double answer = 0;
  if(isDense(source)) {
    uint32_t totalSize = source.totalSize();
    double* iterator = source.data + source.initial_offset;
    for(uint32_t i=0; i<totalSize; i++) {
      answer += iterator[i];
    }
  } else {
    TensorIterator iterator(source);
    do {
      answer += iterator.get();
    } while(iterator.next());
  }
  return answer;
}

} //namespace tensor

