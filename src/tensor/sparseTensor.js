/* jshint esversion: 6 */

var denseTensor = require('./denseTensor');
var tensorUtil = require('./tensorUtil');


class SparseVector {
  constructor(data, length) {
    this.length = length;
    this.data = data;

    if(this.data === undefined)
      this.data = new Map();
    if(Symbol.iterator in this.data)
      this.data = new Map(this.data);

    this.sparse = true;
    this.shape = [length];

    this.numDimensions = 1;
  }

  setLength(length) {
    if(isNaN(length))
      length = undefined;
    this.length = length;
    this.shape = [length];
  }

  totalSize() {
    return this.length;
  }

  at(coord) {
    if(coord instanceof Array)
      coord = coord[0];
    return this.data.get(coord) || 0;
  }

  broadcastAt(coord) {
    if(coord instanceof Array)
      coord = coord[0];
    return this.data.get(coord) || 0;
  }

  set(coord, value) {
    if(coord instanceof Array)
      coord = coord[0];
    if(value === 0)
      this.data.delete(coord);
    else
      this.data.set(coord, value);
  }

  dot(other) {
    return dot(this, other);
  }

  matMul(other, dest) {
    return matMul(this, other, dest);
  }

  clone() {
    return new SparseVector(this.data, this.length);
  }

  applyInPlace(func) {
    for(let [key, value] of this.data) {
      this.data.set(key, func(value));
    }
    return this;
  }

  copyFrom(other) {
    this.length = other.length;
    this.data = new Map(other.data);
  }

  apply(func, dest) {
    if(dest === undefined) {
      dest = this.clone();
      return dest.applyInPlace(func);
    } else {
      for(let [key, value] of this.data) {
        dest.set([key], func(value));
      }
      return dest;
    }
  }

  copyTo(dest) {
    for(let [key, value] of this.data) {
      dest.set([key], value);
    }
  }

  applyBinary(func, other, dest) {
    if(dest === undefined) {
      dest = this.clone();
      return dest.applyBinaryInPlace(func, other);
    } else {
      for(let [key, value] of this.data) {
        dest.set([key], func(value, other.at([key])));
      }
      return dest;
    }
  }

  applyBinaryInPlace(func, other) {
    for(let [key, value] of this.data) {
      this.data.set(key, func(value, other.at(key)));
    }
    return this;    
  }

  toDense(length) {
    if(length === undefined)
      length = this.length;
    if(this.length === undefined)
      length = this.data.size;

    var T = new denseTensor.Tensor({shape: [length]});
    for(let [key, value] of this.data) {
      T.set(key, value);
    }
    return T;
  }
}
exports.SparseVector = SparseVector;

function multiplyScale(sparse, dense, scaleFactor, dest) {
  if(!isNaN(dense)) {
    return scale(sparse, dense*scaleFactor, dest);
  }
  if(dest === undefined) {
    dest = new SparseVector();
    dest.setLength(sparse.length);
  }
  for(let [key, value] of sparse.data) {
    dest.set(key, scaleFactor * value * dense.broadcastAt(key));
  }
  return dest;
}
exports.multiplyScale = multiplyScale;

function addScale(source1, source2, scale1, scale2, dest) {
  if(source2.sparse) {
    return addScaleSparseSparse(source1, source2, scale1, scale2, dest);
  } else {
    return addScaleSparseDense(source1, source2, scale1, scale2, dest);
  }
}
exports.addScale = addScale;

function addScaleSparseSparse(sparse1, sparse2, scale1, scale2, dest) {
  if(dest === undefined) {
    dest = new SparseVector();
    dest.setLength(Math.max(sparse1.length, sparse2.length));
  }
  for(let [key, value] of sparse1.data) {
    dest.set(key, dest.at(key) + scale1 * value);
  }
  for(let [key, value] of sparse2.data) {
    dest.set(key, dest.at(key) + scale2 * value);
  }
  return dest;
}

function addScaleSparseDense(sparse, dense, scale1, scale2, dest) {
  dense = denseTensor.numberToTensor(dense);
  if(dest === undefined) {
    dest = new denseTensor.Tensor({shape: sparse.shape});
  }
  denseTensor.addScale(dest, dense, 0, scale2, dest);
  // denseTensor.scale(dense, scale2, dest);
  for(let [key, value] of sparse.data) {
    dest.set(key, scale1 * value +  dense.broadcastAt(key));
  }
  return dest;
}


function divideScale(sparse, dense, scale, dest) {
  if(dest === undefined) {
    dest = new SparseVector();
    dest.setLength(sparse.length);
  }
  for(let [key, value] of sparse.data) {
    dest.set(key, scale * value / dense.broadcastAt(key));
  }
  return dest;
}
exports.divideScale = divideScale;

function scale(sparse, scaleFactor, dest) {
  if(dest === undefined) {
    dest = new SparseVector();
    dest.setLength(sparse.length);
  }
  for(let [key, value] of sparse.data) {
    dest.set(key, scaleFactor * value);
  }
  return dest; 
}
exports.scale = scale;

function sum(sparse) {
  var answer = 0;
  for(let [key, value] of sparse.data) {
    answer += value;
  }
  return answer;
}
exports.sum = sum;

function dot(sparse, other, dest) {
  if(!sparse.sparse) {
    throw new Error('Attempted to use sparse dot product with non-sparse argument!');
  }
  var product = 0;
  for(let [key, value] of sparse.data) {
    product += value * other.at(key);
  }
  if(dest === undefined) {
    dest = new denseTensor.Tensor([product]);
  } else {
    dest.set([0], product);
  }
  return dest;
}
exports.dot = dot;

function matMul(sparse, other, dest) {
  if(!sparse.sparse) {
    throw new Error('Attempted to use sparse matMul with non-sparse argument!');
  }
  if(dest === undefined) {
    dest = new denseTensor.Tensor({shape: [other.shape[1]]});
  }
  if(other.numDimensions != 2 || dest.numDimensions != 1) {
    throw new Error('DimensionMismatchError');
  }
  for(let i=0; i<dest.shape[0]; i++) {
    let product = 0;
    for(let [key, value] of sparse.data) {
      product += value * other.at([key, i]);
    }
    dest.set([i], product);
  }
  return dest;
}
exports.matMul = matMul;
