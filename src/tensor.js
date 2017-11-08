/*jshint esversion: 6 */

var nodetensor = require('../build/Release/tensor');

DimensionType = Uint32Array;
DataStorageType = Float64Array;
StrideType = Uint32Array;

class Tensor {
  constructor(opts) {
    var {dimensions, numDimensions, strides, initial_offset, data} = opts;
    if(dimensions !== undefined) {
      if(dimensions instanceof Array) {
        dimensions = new DimensionType(dimensions);
      }

      if(numDimensions === undefined) {
        numDimensions = dimensions.length;
      }
      var totalSize = dimensions.reduce((x,y) => {return x*y;});

      if(strides === undefined) {
        strides = new StrideType(numDimensions);
        var currentStride = 1;
        for(let i=0; i<numDimensions; i++) {
          strides[i] = currentStride;
          currentStride *= dimensions[i];
        }
      }
      if(strides instanceof Array) {
        strides = new StrideType(strides);
      }

      if(initial_offset === undefined) {
        initial_offset = 0;
      }

      if(data instanceof Array) {
        data = new DataStorageType(data);
      }

      if(data === undefined) {
        data = new DataStorageType(totalSize);
      }
    } else {
      dimensions = null;
      numDimensions = 0;
      strides = null;
      data = null;
      initial_offset = 0;
    }

    this.dimensions = dimensions;
    this.numDimensions = numDimensions;
    this.strides = strides;
    this.initial_offset = initial_offset;
    this.data = data;
  }

  at(coords) {
    var offset = this.initial_offset;
    for(let i=0; i<this.numDimensions; i++) {
      if(coords[i] >= this.dimensions[i] || coords[i]<0) {
        throw 'coordinate out of range! coord:' + coords[i] + ' dimension: ' + this.dimensions[i];
      }
      offset += coords[i] * this.strides[i];
    }
    return this.data[offset];
  }

  totalSize() {
    return this.dimensions.reduce((x,y) => {return x*y;});
  }

  clone() {
    var opts = {};
    opts.dimensions = this.dimensions;
    opts.numDimensions = this.numDimensions;
    opts.strides = this.strides;
    initial_offset = this.initial_offset;
    opts.data = this.data.slice(0);
    return new Tensor(opts);
  }

  compacted() {
    var data = new DataStorageType(this.totalSize());
    var dimensions = this.dimensions.slice(0);

    var compactified = new Tensor({data, dimensions});

    nodetensor.scale(this, compactified, 1);

    return compactified;
  }

  contract(otherTensor, dimsToContract, dest) {
    return contract(this, otherTensor, dimsToContract, dest);
  }

  matMul(otherTensor, dest) {
    return matMul(otherTensor, dest);
  }

}
exports.Tensor = Tensor;

function print2DTensor(tensor) {
  var strings = [];
  for(let i=0; i<2; i++) {
    for(let j=0; j<2; j++) {
      strings.push(tensor.at([i,j]));
      strings.push(' ');
    }
    strings.push('\n');
  }
  return strings.join('');
}
exports.print2DTensor = print2DTensor;


function zerosLike(dimensions) {
  if(source instanceof Tensor) {
    dimensions = source.dimensions;
  }
  return new Tensor({dimensions});
}
exports.zerosLike = zerosLike;


function broadcastShape(tensor1, tensor2) {
  var numDimensions = max(tensor1.numDimensions, tensor2.numDimensions);
  var dimensions = new DimensionType(numDimensions);
  for(let i=0; i<numDimensions; i++) {
    if(i>=tensor1.numDimensions)
      dimensions[numDimensions - i -1] = tensor2.dimensions[i];
    else if(i>=tensor2.numDimensions)
      dimensions[numDimensions - i -1] = tensor1.dimensions[i];
    else
      dimensions[numDimensions - i -1] = max(tensor1.dimensions[i], tensor2.dimensions[i]);
  }
  return dimensions;
}
exports.broadcastShape = broadcastShape;

function contract(source1, source2, dimsToContract, dest) {
  if(dest === undefined) {
    let dimensions = [];
    for(let dim of source1.dimensions.slice(0,-dimsToContract)) {
      dimensions.push(dim);
    }
    for(let dim of source2.dimensions.slice(dimsToContract)) {
      dimensions.push(dim);
    }
    dest = new Tensor({dimensions});
  }

  nodetensor.contract(source1, source2, dest, dimsToContract);
  return dest;
}
exports.contract = contract;

function matMul(source1, source2, dest) {
  return contract(source1, source2, 1, dest);
}
exports.matMul = matMul;

function numberToTensor(number) {
  return new Tensor({dimensions:[1], data: new DataStorageType([number])});
}
exports.numberToTensor = numberToTensor;



