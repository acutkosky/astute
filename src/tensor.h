


#define ASSERT(expr, error) if(!(expr)) { return error; }
#define ASSERT_SIZE_EQUAL_3(a, b, c)  ASSERT(a.totalSize() == b.totalSize() && a.totalSize() == c.totalSize(), SizeMismatchError)
#define MIN(a,b) (a>b?b:a)
#define MAX(a,b) (a>b?a:b)


enum TensorErrors {
  NoError = 0,
  SizeMismatchError = 1,
  DimensionMismatchError = 2
};


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
  bool dimensionInReversedOrder;

  int totalSize(void);

  double& at(int* coords);

  double& at(int* prefixCoords, int* suffixCoords, int suffixSize);

  double& broadcast_at(int* coords);

  void transpose_(void);
};

struct broadcastIterator {
  int numDimensions;
  int* dimensions;
  int* coords;
  bool ended;

  broadcastIterator(Tensor& t1, Tensor& t2) {
    numDimensions = MAX(t1.numDimensions, t2.numDimensions);
    dimensions = new int[numDimensions];
    coords = new int[numDimensions];
    ended = false;

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

    for(int i=0; i<numDimensions; i++) {
      dimensions[i] = MAX(t1.dimensions[coordIndex1], t2.dimensions[coordIndex2]);
      coordIndex1 += coordIndexIncrement1;
      coordIndex2 += coordIndexIncrement2;
      coords[i] = 0.0;
    }
  }

  broadcastIterator(Tensor& t) {
    numDimensions = t.numDimensions;
    dimensions = new int[numDimensions];
    coords = new int[numDimensions];
    ended = false;

    int coordIndex = 0;
    int coordIndexIncrement = 1;
    if(!t.dimensionInReversedOrder) {
      coordIndexIncrement = -1;
      coordIndex = t.numDimensions - 1;
    }

    for(int i=0; i<numDimensions; i++) {
      dimensions[i] = t.dimensions[coordIndex];
      coordIndex += coordIndexIncrement;
      coords[i] = 0.0;
    }
  }

  broadcastIterator(Tensor& t, int dimensionsToKeep) {
    numDimensions = MAX(dimensionsToKeep, -dimensionsToKeep);

    dimensions = new int[numDimensions];
    coords = new int[numDimensions];
    ended = false;

    int coordIndex = 0;
    int coordIndexIncrement = 1;
    if(!t.dimensionInReversedOrder) {
      coordIndexIncrement = -1;
      coordIndex = t.numDimensions - 1;
    }

    if(dimensionsToKeep < 0) {
      coordIndex += (t.numDimensions - numDimensions) * coordIndexIncrement;
    }

    for(int i=0; i<numDimensions; i++) {
      dimensions[i] = t.dimensions[coordIndex];
      coordIndex += coordIndexIncrement;
      coords[i] = 0.0;
    }
  }

  void reset(void) {
    ended = false;
    for(int i=0; i<numDimensions; i++) {
      coords[i] = 0.0;
    }
  }

  ~broadcastIterator() {
    delete [] dimensions;
    delete [] coords;
  }

  void next(void);
};

int addScale(Tensor& source1, Tensor& source2, double scale1, double scale2, Tensor& dest);

int multiplyScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest);

int divideScale(Tensor& source1, Tensor& source2, double scale, Tensor& dest);

int add(Tensor& source1, Tensor& source2, Tensor& dest);

int subtract(Tensor& source1, Tensor& source2, Tensor& dest);

int multiply(Tensor& source1, Tensor& source2, Tensor& dest);

int divide(Tensor& source1, Tensor& source2, Tensor& dest);

int matMul(Tensor& source1, Tensor& source2, int sumDims, Tensor& dest);

int scalarProduct(Tensor& source1, Tensor& source2, double& product);

