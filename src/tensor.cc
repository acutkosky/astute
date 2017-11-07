#include <iostream>

#include "tensor.h"
namespace tensor {

TensorError globalError;

using std::cout;
using std::endl;

uint32_t Tensor::totalSize(void) {
  if(this->numDimensions == 0) {
    return 0;
  }
  uint32_t accumulator = 1;
  for(uint32_t i=0; i<this->numDimensions; i++) {
    accumulator *= this->dimensions[i];
  }
  return accumulator;
}

uint32_t Tensor::maximumOffset(void) {
  if(this->numDimensions == 0) {
    return this->initial_offset;
  }
  uint32_t offset = this->initial_offset;

  for(uint32_t i=0; i<this->numDimensions; i++) {
    offset += this->dimensions[i]*this->strides[i];
  }

  return offset;
}

bool Tensor::isValid(void) {
  return this->data != NULL;
}

double& Tensor::at(uint32_t* coords, TensorError* error) {
  uint32_t offset = this->initial_offset;
  for(uint32_t i=0; i<this->numDimensions; i++) {
    if(coords[i] >= this->dimensions[i]) {
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
    uint32_t dimension = this->dimensions[this->numDimensions - i - 1];
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
  * dimensionsInReversedOrder specifies whether the tensor is 
  * stored in "column major" or "row major" order. Specifically, if
  * dimensionsInReversedOrder is true, the ijk th element of a NxMxK tensor
  * is located at
  *   data[i + j*N + k*MN]
  * and so strides will be set to 
  *   [1, N, MN]
  * if dimensionsInReversedOrder is true, the ikj element is at
  *   data[i*MK + j*K + k]
  * and so strides will be set to
  * [MK, K, 1]
  *
  * It is not the business of this library to manage memory that might
  * be used externally; we assume that the strides array already exists.
  **/

void Tensor::setStrides(bool dimensionsInReversedOrder) {
  if(dimensionsInReversedOrder) {
    uint32_t currentStride = 1;
    for(uint32_t i=0; i<numDimensions; i++) {
      this->strides[i] = currentStride;
      currentStride *= this->dimensions[i];
    }
  } else {
    uint32_t currentStride = 1;
    for(int i=this->numDimensions-1; i>=0; i--) {
      this->strides[i] = currentStride;
      currentStride *= this->dimensions[i];
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
    dest.dimensions[i] = source.dimensions[source.numDimensions-1-i];
    dest.strides[i] = source.strides[source.numDimensions-1-i];
  }
  dest.initial_offset = source.initial_offset;
}


bool matchedDimensions(Tensor& t1, Tensor& t2) {
  if(t1.numDimensions != t2.numDimensions)
    return false;

  for(uint32_t i=0; i<t1.numDimensions; i++) {
    if(t1.dimensions[i] != t2.dimensions[i])
      return false;
  }

  return true;
}

bool compatibleDimensions(Tensor& t1, Tensor& t2) {
  uint32_t minimumDim = MIN(t1.numDimensions, t2.numDimensions);

  for(uint32_t i=0; i<minimumDim; i++) {
    if(t1.dimensions[t1.numDimensions-1-i] == 1)
      continue;
    if(t2.dimensions[t2.numDimensions-1-i] == 1)
      continue;
    if(t1.dimensions[t1.numDimensions-1-i] == t2.dimensions[t2.numDimensions-1-i])
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
      dest.dimensions[destDimensionIndex] = source.dimensions[i];
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
  uint32_t i = 0;
  uint32_t offset = 0;
  currentCoords[i] = (currentCoords[i] + 1) % T->dimensions[i];

  offset += T->strides[i];
  while(currentCoords[i] == 0) {
    offset -= T->strides[i] * T->dimensions[i];
    i++;
    if(i>=T->numDimensions) {
      ended = true;
      break;
    }

    currentCoords[i] = (currentCoords[i] + 1) % T->dimensions[i];
    offset += T->strides[i];

   }
   iterator += offset;
  return !ended;
}

double& TensorIterator::get(void) {
  return *iterator;
}


bool MultiIndexIterator::next(void) {
  uint32_t i = 0;
  currentCoords[i] = (currentCoords[i] + 1) % dimensions[i];
  while(currentCoords[i] == 0) {
    i++;
    if(i>=numDimensions) {
      ended = true;
      break;
    }
    currentCoords[i] = (currentCoords[i] + 1) % dimensions[i];
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

bool compatibleForContraction(Tensor& source1, Tensor& source2, uint32_t dimsToContract) {

  if(source1.numDimensions <= dimsToContract)
    return false;
  if(source2.numDimensions <= dimsToContract)
    return false;


  for(uint32_t i=0; i<dimsToContract; i++) {
    uint32_t source1Offset = source1.numDimensions - 1 - i;
    uint32_t source2Offset = i;
    if(source1.dimensions[source1Offset] != source2.dimensions[source2Offset]) 
      return false;
  }

  return true;
}

void contract(Tensor& source1, Tensor& source2, Tensor& dest, uint32_t dimsToContract, TensorError* error) {

  if(dest.numDimensions != 
      source1.numDimensions + source2.numDimensions - 2 * dimsToContract) {
    *error = DimensionMismatchError;
    return;
  }

  if(!compatibleForContraction(source1, source2, dimsToContract)) {
    *error = DimensionMismatchError;
    return;
  }

  MultiIndexIterator destIterator(dest.dimensions, dest.numDimensions);

  uint32_t* dimRange = new uint32_t[dest.numDimensions];
  for(uint32_t i=0; i<dest.numDimensions; i++) {
    dimRange[i] = i;
  }
  Tensor sub1, sub2;

  sub1.dimensions = new uint32_t[dimsToContract];
  sub1.strides = new uint32_t[dimsToContract];
  sub1.numDimensions = dimsToContract;
  sub1.data = source1.data;

  sub2.dimensions = new uint32_t[dimsToContract];
  sub2.strides = new uint32_t[dimsToContract];
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

  delete [] sub1.dimensions;
  delete [] sub1.strides;
  delete [] sub2.dimensions;
  delete [] sub2.strides;
}

bool isBroadcastDimension(Tensor& source1, Tensor& source2, Tensor& dest) {
  uint32_t maxDimensions = MAX(source1.numDimensions, source2.numDimensions);
  if(dest.numDimensions != maxDimensions)
    return false;

  uint32_t dimension1, dimension2;

  for(uint32_t i=0; i<maxDimensions; i++) {
    dimension1 = dimension2 = 1;
    if(i<source1.numDimensions)
      dimension1 = source1.dimensions[source1.numDimensions - i -1];

    if(i<source2.numDimensions)
      dimension2= source2.dimensions[source2.numDimensions - i -1];

    if(dest.dimensions[maxDimensions - i -1] != MAX(dimension1, dimension2))
      return false;
  }
  return true;
}

void addScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale1, double scale2, TensorError* error) {
  if(!compatibleDimensions(source1, source2)) {
    *error = DimensionMismatchError;
    return;
  }
  if(!isBroadcastDimension(source1, source2, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  MultiIndexIterator destIterator(dest.dimensions, dest.numDimensions);
  uint32_t numDim = dest.numDimensions;
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale1 * source1.broadcast_at(currentCoords, numDim) +
      scale2 * source2.broadcast_at(currentCoords, numDim);
  } while(destIterator.next());

}

void multiplyScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale, TensorError* error) {
  if(!compatibleDimensions(source1, source2)) {
    *error = DimensionMismatchError;
    return;
  }
  if(!isBroadcastDimension(source1, source2, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  MultiIndexIterator destIterator(dest.dimensions, dest.numDimensions);
  uint32_t numDim = dest.numDimensions;
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale * 
      source1.broadcast_at(currentCoords, numDim) *
      source2.broadcast_at(currentCoords, numDim);
  } while(destIterator.next());

}

void divideScale(Tensor& source1, Tensor& source2, Tensor& dest, double scale, TensorError* error) {
  if(!compatibleDimensions(source1, source2)) {
    *error = DimensionMismatchError;
    return;
  }
  if(!isBroadcastDimension(source1, source2, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  MultiIndexIterator destIterator(dest.dimensions, dest.numDimensions);
  uint32_t numDim = dest.numDimensions;
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale * 
      source1.broadcast_at(currentCoords, numDim) /
      source2.broadcast_at(currentCoords, numDim);
  } while(destIterator.next());

}

void add(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return addScale(source1, source2, dest, 1, 1, error);
}

void subtract(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return addScale(source1, source2, dest, 1, -1, error);
}

void multiply(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return multiplyScale(source1, source2, dest, 1, error);
}

void divide(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return divideScale(source1, source2, dest, 1, error);
}

void scale(Tensor& source, Tensor& dest, double scale, TensorError* error) {
  if(!matchedDimensions(source, dest)) {
    *error = DimensionMismatchError;
    return;
  }

  MultiIndexIterator destIterator(dest.dimensions, dest.numDimensions);
  do {
    uint32_t* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale * source.at(currentCoords);
  } while(destIterator.next());
}

void matMul(Tensor& source1, Tensor& source2, Tensor& dest, TensorError* error) {
  return contract(source1, source2, dest, 1, error);
}

} //namespace tensor

