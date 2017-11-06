#include <iostream>

#include "tensor.h"
namespace tensor {

TensorError globalError;

using std::cout;
using std::endl;

int Tensor::totalSize(void) {
  if(this->numDimensions == 0) {
    return 0;
  }
  int accumulator = 1;
  for(int i=0; i<this->numDimensions; i++) {
    accumulator *= this->dimensions[i];
  }
  return accumulator;
}


double& Tensor::at(int* coords, TensorError* error) {
  int offset = this->initial_offset;
  for(int i=0; i<this->numDimensions; i++) {
    if(coords[i] < 0 || coords[i] >= this->dimensions[i]) {
      *error = IndexOutOfBounds;
      return this->data[offset];
    }
    offset += coords[i] * this->strides[i];
  }
  return this->data[offset];
}

double& Tensor::broadcast_at(int* coords, int numCoords, TensorError* error) {
  int offset = this->initial_offset;
  for(int i=0; i<this->numDimensions; i++) {
    int dimension = this->dimensions[this->numDimensions - i - 1];
    int stride = this->strides[this->numDimensions - i - 1];
    int coord = coords[numCoords -i -1];
    if(coord < 0) {
      *error = IndexOutOfBounds;
      return this->data[offset];
    }
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
  * sets the strides pointer in the tensor object.
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
    int currentStride = 1;
    for(int i=0; i<numDimensions; i++) {
      this->strides[i] = currentStride;
      currentStride *= this->dimensions[i];
    }
  } else {
    int currentStride = 1;
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
  for(int i=0; i<source.numDimensions; i++) {
    dest.dimensions[i] = source.dimensions[source.numDimensions-1-i];
    dest.strides[i] = source.strides[source.numDimensions-1-i];
  }
  dest.initial_offset = source.initial_offset;
}


bool matchedDimensions(Tensor& t1, Tensor& t2) {
  if(t1.numDimensions != t2.numDimensions)
    return false;

  for(int i=0; i<t1.numDimensions; i++) {
    if(t1.dimensions[i] != t2.dimensions[i])
      return false;
  }

  return true;
}

bool compatibleDimensions(Tensor& t1, Tensor& t2) {
  int minimumDim = MIN(t1.numDimensions, t2.numDimensions);

  for(int i=0; i<minimumDim; i++) {
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
void subTensor(Tensor& source, int* heldCoords, int* heldValues, int numHeld, Tensor& dest, TensorError* error) {
  if(dest.data!=NULL && dest.data!=source.data) {
    *error = MemoryLeakError;
    return;   
  }
  if(dest.numDimensions != source.numDimensions - numHeld) {
    *error = DimensionMismatchError;
    return;
  }
  int heldCoordsIndex = 0;
  int destDimensionIndex = 0;
  int offset = source.initial_offset;
  for(int i=0; i<source.numDimensions; i++) {
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
  dest.data = source.data;
}

bool TensorIterator::next(void) {
  int i = 0;
  int offset = 0;
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
  int i = 0;
  int offset = 0;
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

int* MultiIndexIterator::get(void) {
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
  int i =0;
  do {
    i++;
    product += iter1.get() * iter2.get();
  } while(iter1.next() && iter2.next());
  return product;

}

void print2DCoord(int* coords) {
  cout<<coords[0]<<" "<<coords[1]<<endl;
}

bool compatibleForContraction(Tensor& source1, Tensor& source2, int dimsToContract) {

  if(source1.numDimensions <= dimsToContract)
    return false;
  if(source2.numDimensions <= dimsToContract)
    return false;


  for(int i=0; i<dimsToContract; i++) {
    int source1Offset = source1.numDimensions-dimsToContract - 1 - i;
    int source2Offset = i;
    if(source1.dimensions[source1Offset] != source2.dimensions[source2Offset]) 
      return false;
  }

  return true;
}

void contract(Tensor& source1, Tensor& source2, Tensor& dest, int dimsToContract, TensorError* error) {

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

  int* dimRange = new int[dest.numDimensions];
  for(int i=0; i<dest.numDimensions; i++) {
    dimRange[i] = i;
  }
  Tensor sub1, sub2;

  sub1.dimensions = new int[dimsToContract];
  sub1.strides = new int[dimsToContract];
  sub1.numDimensions = dimsToContract;

  sub2.dimensions = new int[dimsToContract];
  sub2.strides = new int[dimsToContract];
  sub2.numDimensions = dimsToContract;

  do {
    int* currentCoords = destIterator.get();
    subTensor(source1, 
              dimRange, 
              currentCoords, 
              source1.numDimensions - dimsToContract,
              sub1,
              error);
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
  int maxDimensions = MAX(source1.numDimensions, source2.numDimensions);
  if(dest.numDimensions != maxDimensions)
    return false;

  int dimension1, dimension2;

  for(int i=0; i<maxDimensions; i++) {
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
  int numDim = dest.numDimensions;
  do {
    int* currentCoords = destIterator.get();
    dest.at(currentCoords) = 
      scale1 * source1.broadcast_at(currentCoords, numDim) +
      scale2 * source2.broadcast_at(currentCoords, numDim);
  } while(destIterator.next());

}

} //namespace tensor

// int main(void) {
//   cout<<"hello world\n";
//   return 0;
// }
/*
bool matchedDimensions(Tensor& t1, Tensor& t2) {
  if(t1.dimensionInReversedOrder != t2.dimensionInReversedOrder)
    return false;

  if(t1.numDimensions != t2.numDimensions)
    return false;

  for(int i=0; i<t1.numDimensions; i++) {
    if(t1.dimensions[i] != t2.dimensions[i])
      return false;
  }

  return true;
}

bool compatibleDimensions(Tensor& t1, Tensor& t2) {
  int minimumDim = MIN(t1.numDimensions, t2.numDimensions);


  int coordIndex1 = 0;
  int coordIndexIncrement1 = 1;
  if(!t1.dimensionInReversedOrder) {
    coordIndexIncrement1 = -1;
    coordIndex1 = t1.numDimensions - 1;
  }

  int coordIndex2= 0;
  int coordIndexIncrement2 = 1;
  if(!t2.dimensionInReversedOrder) {
    coordIndexIncrement2 = -1;
    coordIndex2 = t2.numDimensions - 1;
  }

  for(int i=0; i<minimumDim; i++) {
    int t1Dim = t1.dimensions[coordIndex1];
    int t2Dim = t2.dimensions[coordIndex2];
    if(t1Dim == t2Dim)
      continue;
    if(t1Dim == 1)
      continue;
    if(t2Dim == 1)
      continue;

    return false;

    coordIndex1 += coordIndexIncrement1;
    coordIndex2 += coordIndexIncrement2;
  }
  return true;
}

void broadcastIterator::next(void) {
  int i = 0;
  coords[i] = (coords[i] + 1) % dimensions[i];
  while(coords[i] == 0) {
    i++;
    if(i>=numDimensions) {
      ended = true;
      break;
    }

    coords[i] = (coords[i] + 1) % dimensions[i];
  }
}

double& Tensor::at(int* coords) {

  int offset = 0;
  int stride = 1;

  int coordIndex = 0;
  int coordIndexIncrement = 1;
  if(!this->dimensionInReversedOrder) {
    coordIndexIncrement = -1;
    coordIndex = this->numDimensions - 1;
  }
  for(int i=0; i<this->numDimensions; i++) {
    offset += stride * coords[coordIndex];
    stride *= this->dimensions[i];
    coordIndex += coordIndexIncrement;
  }

  return this->data[offset];
}

double& Tensor::at(int* prefixCoords, int* suffixCoords, int prefixSize) {

  int offset = 0;
  int stride = 1;

  int coordIndex = 0;
  int coordIndexIncrement = 1;
  double coord;
  if(!this->dimensionInReversedOrder) {
    coordIndexIncrement = -1;
    coordIndex = this->numDimensions - 1;
  }

  for(int i=0; i<this->numDimensions; i++) {
    if(this->dimensionInReversedOrder) {
      if(i<prefixSize)
        coord = prefixCoords[coordIndex];
      else
        coord = suffixCoords[coordIndex - prefixSize];
    } else {
      if(i<this->numDimensions - prefixSize)
        coord = suffixCoords[coordIndex- prefixSize];
      else
        coord = prefixCoords[coordIndex];     
    }
    offset += stride * coord;
    stride *= this->dimensions[i];
    coordIndex += coordIndexIncrement;
  }

  return this->data[offset];

}

double& Tensor::broadcast_at(int* coords) {
  int offset = 0;
  int stride = 1;

  int coordIndex = 0;
  int coordIndexIncrement = 1;
  if(!this->dimensionInReversedOrder) {
    coordIndexIncrement = -1;
    coordIndex = this->numDimensions - 1;
  }

  for(int i=0; i<this->numDimensions; i++) {

    int coord = coords[coordIndex];
    if(this->dimensions[i]==1)
      coord = 0;

    offset += stride * coord;
    stride *= this->dimensions[i];
    coordIndex += coordIndexIncrement;
  }

  return this->data[offset];
}

bool is_prefix(Tensor& t1, Tensor& t2) {

  if(t1.dimensionInReversedOrder != t2.dimensionInReversedOrder)
    return false;


  int minimumDim = MIN(t1.numDimensions, t2.numDimensions);

  if(t1.dimensionInReversedOrder) {
    for(int i=0; i<minimumDim; i++) {
      if(t1.dimensions[i] != t2.dimensions[i])
        return false;
    }
  } else {
    for(int i=0; i<minimumDim; i++) {
      if(t1.dimensions[t1.numDimensions-1 - i] != t2.dimensions[t2.numDimensions-1 -i])
        return false;
    }
  }
  return true;
}

int addScale(Tensor& source1, Tensor& source2, double scale1, double scale2, Tensor& dest) {

  if(matchedDimensions(source1, source2)) {
    cout<<dest.totalSize()<<endl;
    cout<<source1.totalSize()<<endl;
    cout<<source2.totalSize()<<endl;
    ASSERT_SIZE_EQUAL_3(source1, source2, dest)


    int totalSize = source1.totalSize();
    for(int i=0; i<totalSize; i++) {
      dest.data[i] = scale1 * source1.data[i] + scale2 * source2.data[i];
    }
    cout<<endl;
    return NoError;
  } else if(is_prefix(source1, source2)) {
    int size1 = source1.totalSize();
    int size2 = source2.totalSize();
    int maxSize = MAX(size1, size2);
    ASSERT(maxSize == dest.totalSize(), SizeMismatchError)

    for(int i=0; i<maxSize; i++) {
      dest.data[i] = scale1 * source1.data[i % size1] + scale2 * source2.data[i & size2];
    }
    return NoError;
  } else if(compatibleDimensions(source1, source2)) {
    broadcastIterator iterator(source1, source2);
    while(!iterator.ended) {
      dest.broadcast_at(iterator.coords) = scale1 * source1.broadcast_at(iterator.coords) + scale2 * source2.broadcast_at(iterator.coords);
      iterator.next();
    }

    return NoError;
  }

  return SizeMismatchError;
}

int multiplyScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest) {


  if(matchedDimensions(source1, source2)) {
    ASSERT_SIZE_EQUAL_3(source1, source2, dest)

    int totalSize = source1.totalSize();
    for(int i=0; i<totalSize; i++) {
      dest.data[i] = scale * source1.data[i] * source2.data[i];
    }
    return NoError;
  } else if(is_prefix(source1, source2)) {
    int size1 = source1.totalSize();
    int size2 = source2.totalSize();
    int maxSize = MAX(size1, size2);
    ASSERT(maxSize == dest.totalSize(), SizeMismatchError)

    for(int i=0; i<maxSize; i++) {
      dest.data[i] = scale * source1.data[i % size1] * source2.data[i & size2];
    }
    return NoError;
  } else if(compatibleDimensions(source1, source2)) {
    broadcastIterator iterator(source1, source2);
    while(!iterator.ended) {
      dest.broadcast_at(iterator.coords) = scale * source1.broadcast_at(iterator.coords) * source2.broadcast_at(iterator.coords);
      iterator.next();
    }

    return NoError;
  }

  return SizeMismatchError;

}

int divideScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest) {
  if(matchedDimensions(source1, source2)) {
    ASSERT_SIZE_EQUAL_3(source1, source2, dest)

    int totalSize = source1.totalSize();
    for(int i=0; i<totalSize; i++) {
      dest.data[i] = scale * source1.data[i] / source2.data[i];
    }
    return NoError;
  } else if(is_prefix(source1, source2)) {
    int size1 = source1.totalSize();
    int size2 = source2.totalSize();
    int maxSize = MAX(size1, size2);
    ASSERT(maxSize == dest.totalSize(), SizeMismatchError)

    for(int i=0; i<maxSize; i++) {
      dest.data[i] = scale * source1.data[i % size1] / source2.data[i & size2];
    }
    return NoError;
  } else if(compatibleDimensions(source1, source2)) {
    broadcastIterator iterator(source1, source2);
    while(!iterator.ended) {
      dest.broadcast_at(iterator.coords) = scale * source1.broadcast_at(iterator.coords) / source2.broadcast_at(iterator.coords);
      iterator.next();
    }

    return NoError;
  }

  return SizeMismatchError;
}

int add(Tensor& source1, Tensor& source2, Tensor& dest) {
  return addScale(source1, source2, 1.0, 1.0, dest);
}


int subtract(Tensor& source1, Tensor& source2, Tensor& dest) {
  return addScale(source1, source2, 1.0, -1.0, dest);
}

int multiply(Tensor& source1, Tensor& source2, Tensor& dest) {
  return multiplyScale(source1, source2, 1.0, dest);
}

int divide(Tensor& source1, Tensor& source2, Tensor& dest) {
  return divideScale(source1, source2, 1.0, dest);
}

double scalarProduct(Tensor& source1, Tensor& source2, double& product) {
  product = 0;


  if(matchedDimensions(source1, source2)) {

    int totalSize = source1.totalSize();
    for(int i=0; i<totalSize; i++) {
      product += source1.data[i] * source2.data[i];
    }
    return 0;
  } else if(is_prefix(source1, source2)) {
    int size1 = source1.totalSize();
    int size2 = source2.totalSize();
    int maxSize = MAX(size1, size2);

    for(int i=0; i<maxSize; i++) {
      product += source1.data[i % size1] * source2.data[i & size2];
    }
    return 0;
  } else if(compatibleDimensions(source1, source2)) {
    broadcastIterator iterator(source1, source2);
    while(!iterator.ended) {
      product += source1.broadcast_at(iterator.coords) * source2.broadcast_at(iterator.coords);
      iterator.next();
    }

    return NoError;
  }

  return SizeMismatchError;

}

int checkMatMulDimensions(Tensor& source1, Tensor& source2, int sumDims, Tensor& dest) {
  ASSERT(source1.numDimensions > sumDims, DimensionMismatchError)
  ASSERT(source2.numDimensions > sumDims, DimensionMismatchError)
  ASSERT(dest.numDimensions = source1.numDimensions + source2.numDimensions - 2*sumDims, DimensionMismatchError)


  int coordIndex1 = 0;
  int coordIndexIncrement1 = 1;
  if(!source1.dimensionInReversedOrder) {
    coordIndexIncrement1 = -1;
    coordIndex1 = source1.numDimensions - 1;
  }

  int coordIndex2= 0;
  int coordIndexIncrement2 = 1;
  if(!source2.dimensionInReversedOrder) {
    coordIndexIncrement2 = -1;
    coordIndex2 = source2.numDimensions - 1;
  }

  int coordIndexDest= 0;
  int coordIndexIncrementDest = 1;
  if(!dest.dimensionInReversedOrder) {
    coordIndexIncrementDest = -1;
    coordIndexDest = dest.numDimensions - 1;
  }

  for(int i=0; i<source1.numDimensions - sumDims; i++) {
    ASSERT(source1.dimensions[coordIndex1] == dest.dimensions[coordIndexDest], DimensionMismatchError)
    coordIndex1 += coordIndexIncrement1;
    coordIndexDest += coordIndexIncrementDest;
  }

  if(sumDims > 1) {
    for(int i=0; i<sumDims; i++) {
      ASSERT(source1.dimensions[coordIndex1] == source2.dimensions[coordIndex2], DimensionMismatchError)
      ASSERT(source1.dimensions[coordIndex1] == dest.dimensions[coordIndexDest], DimensionMismatchError)
      coordIndex1 += coordIndexIncrement1;
      coordIndex2 += coordIndexIncrement2;
      coordIndexDest += coordIndexIncrementDest;
    }
  }

  for(int i=sumDims; i<source2.numDimensions; i++) {
    cout<<"i "<<i<<" coordIndex2: "<<coordIndex2<<" coorddestIndex: "<<coordIndexDest<<endl;
    ASSERT(source2.dimensions[coordIndex2] == dest.dimensions[coordIndexDest], DimensionMismatchError)
    coordIndex2 += coordIndexIncrement2;
    coordIndexDest += coordIndexIncrementDest;
  }

  return NoError;
}

void coutcoord(int* coords, int numcoords) {
  for(int i=0; i<numcoords; i++) {
    cout<<coords[i]<<" ";
  }
}

int sumDimension(Tensor &source, int dimension, Tensor &dest) {

}

int matMul(Tensor& source1, Tensor& source2, int sumDims, Tensor& dest) {
  ASSERT(checkMatMulDimensions(source1, source2, sumDims, dest)==NoError, DimensionMismatchError)


  broadcastIterator innerIterator(source2, sumDims);
  broadcastIterator source1Iterator(source1, source1.numDimensions - sumDims);
  broadcastIterator source2Iterator(source2, sumDims - source2.numDimensions);
  while(!source2Iterator.ended) {
    while(!source1Iterator.ended) {
      double value = 0.0;
      while(!innerIterator.ended) {
        ASSERT(!source2Iterator.ended, DimensionMismatchError)
        cout<<"dest ";
        coutcoord(source1Iterator.coords, 1);
        coutcoord(source2Iterator.coords, 1);
        cout<<" += ";
        coutcoord(source1Iterator.coords, 1);
        coutcoord(innerIterator.coords, 1);
        cout<<" * ";
        coutcoord(source2Iterator.coords, 1);
        coutcoord(innerIterator.coords, 1);
        cout<<endl;
        cout<<source1.at(source1Iterator.coords, innerIterator.coords, source1.numDimensions-sumDims)<<"  * "<<source2.at(innerIterator.coords, source2Iterator.coords, sumDims)<<endl;

        value += source1.at(source1Iterator.coords, innerIterator.coords, source1.numDimensions-sumDims) * source2.at(innerIterator.coords, source2Iterator.coords, sumDims);
        innerIterator.next();
      }
      dest.at(source1Iterator.coords, source2Iterator.coords, source1.numDimensions-sumDims) = value;
      innerIterator.reset();
      source1Iterator.next();
    }
    source1Iterator.reset();
    source2Iterator.next();
  }


  return NoError;
}

void coutmatrix(Tensor& t) {
  int c[] = {0,0};
  for(int i=0;i<2;i++) {
    for(int j=0;j<2;j++) {
      c[0] = i;
      c[1] = j;
      cout<<t.at(c)<<" ";
    }
    cout<<endl;
  }
}

int main(void) {
  Tensor t1, t2, t3;
  t1.numDimensions = 2;
  t2.numDimensions = 2;
  t3.numDimensions = 2;

  t1.dimensionInReversedOrder = true;
  t2.dimensionInReversedOrder = true;
  t3.dimensionInReversedOrder = true;

  t1.dimensions = new int[2];
  t2.dimensions = new int[2];
  t3.dimensions = new int[2];

  t1.dimensions[0] = 2;
  t1.dimensions[1] = 2;

  t2.dimensions[0] = 2;
  t2.dimensions[1] = 2;

  t3.dimensions[0] = 2;
  t3.dimensions[1] = 2;

  t1.data = new double[4];
  t2.data = new double[4];
  t3.data = new double[4];


  for(int i=0; i<4;i++) {
    t1.data[i] = i+1;
    t2.data[i] = i+2;
    t3.data[i] = 0;
  }

  int ret = matMul(t1, t2, 1, t3);


  int c[] = {0,1};

  int d[] = {0,1};
  int SizeMismatch = 5;
  printf("return value: %d\n", ret);
  coutmatrix(t1);
  cout<<endl;
  coutmatrix(t2);
  cout<<endl;
  coutmatrix(t3);
  cout<<endl;
  printf("first coord: %g\n", t1.at(d));
  printf("first coord: %g\n", t2.at(c));
  printf("first coord: %g\n", t3.at(c));
  printf("enum: %d\n",ret );

  return 0;

}

*/
