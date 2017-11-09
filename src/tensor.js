/*jshint esversion: 6 */

var nodetensor = require('../build/Release/tensor');

var DimensionType = Uint32Array;
var DataStorageType = Float64Array;
var StrideType = Uint32Array;

class Tensor {
  constructor(opts) {
    var {shape, numDimensions, strides, initial_offset, data} = opts;
    if(shape !== undefined) {
      if(shape instanceof Array) {
        shape = new DimensionType(shape);
      }

      if(numDimensions === undefined) {
        numDimensions = shape.length;
      }
      var totalSize = shape.reduce((x,y) => {return x*y;});

      if(strides === undefined) {
        strides = new StrideType(numDimensions);
        var currentStride = 1;
        for(let i=0; i<numDimensions; i++) {
          strides[i] = currentStride;
          currentStride *= shape[i];
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
      shape = null;
      numDimensions = 0;
      strides = null;
      data = null;
      initial_offset = 0;
    }

    this.shape = shape;
    this.numDimensions = numDimensions;
    this.strides = strides;
    this.initial_offset = initial_offset;
    this.data = data;
  }

  at(coords) {
    var offset = this.initial_offset;
    for(let i=0; i<this.numDimensions; i++) {
      if(coords[i] >= this.shape[i] || coords[i]<0) {
        throw 'coordinate out of range! coord:' + coords[i] + ' dimension: ' + this.shape[i];
      }
      offset += coords[i] * this.strides[i];
    }
    return this.data[offset];
  }

  totalSize() {
    return this.shape.reduce((x,y) => {return x*y;});
  }

  clone() {
    var opts = {};
    opts.shape = this.shape;
    opts.numDimensions = this.numDimensions;
    opts.strides = this.strides;
    initial_offset = this.initial_offset;
    opts.data = this.data.slice(0);
    return new Tensor(opts);
  }

  compacted() {
    var data = new DataStorageType(this.totalSize());
    var shape = this.shape.slice(0);

    var compactified = new Tensor({data, shape});

    nodetensor.scale(this, compactified, 1);

    return compactified;
  }

  contract(otherTensor, dimsToContract, dest) {
    return contract(this, otherTensor, dimsToContract, dest);
  }

  matMul(otherTensor, dest) {
    return matMul(this, otherTensor, dest);
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


function zerosLike(shape) {
  if(source instanceof Tensor) {
    shape = source.shape;
  }
  return new Tensor({shape});
}
exports.zerosLike = zerosLike;


function broadcastShape(tensor1, tensor2) {
  var numDimensions = max(tensor1.numDimensions, tensor2.numDimensions);
  var shape = new DimensionType(numDimensions);
  for(let i=0; i<numDimensions; i++) {
    if(i>=tensor1.numDimensions)
      shape[numDimensions - i -1] = tensor2.shape[i];
    else if(i>=tensor2.numDimensions)
      shape[numDimensions - i -1] = tensor1.shape[i];
    else
      shape[numDimensions - i -1] = max(tensor1.shape[i], tensor2.shape[i]);
  }
  return shape;
}
exports.broadcastShape = broadcastShape;


function contract(source1, source2, dimsToContract, dest) {
  if(dest === undefined) {
    let shape = [];
    for(let dim of source1.shape.slice(0,-dimsToContract)) {
      shape.push(dim);
    }
    for(let dim of source2.shape.slice(dimsToContract)) {
      shape.push(dim);
    }
    dest = new Tensor({shape});
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
  if(number instanceof Tensor)
    return number;

  return new Tensor({shape:[1], data: new DataStorageType([number])});
}
exports.numberToTensor = numberToTensor;



function exportBinaryOp(opname) {
  function binaryOp(source1, source2, dest) {
    source1 = numberToTensor(source1);
    source2 = numberToTensor(source2);
    if(dest === undefined)
      dest = zerosLike(broadcastShape(source1, source2));

    nodetensor[opname](souce1, source2, dest);

    return dest;
  }
  return binaryOp;
}



function pow(base, power, dest) {
  power = numberToTensor(power);
  base = numberToTensor(base);

  if(dest === undefined)
    dest = zerosLike(broadcastShape(base, power));

  nodetensor.exp(base, power, dest);

  return dest;
}
exports.pow = pow;

function fmod(source, mod, dest) {
  source = numberToTensor(source);
  mod = number(mod);
  if(dest === undefined)
    dest = zerosLike(broadcastShape(source,  mode));

  nodetensor.fmod(source, mod, dest);

  return dest;
}
exports.fmod = fmod;

