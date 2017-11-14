/* jshint esversion: 6 */

var tensorBinding = require('../build/Release/tensorBinding');
var tensorUtil = require('./tensorUtil');

var DimensionType = Uint32Array;
var DataStorageType = Float64Array;
var StrideType = Uint32Array;

function parseArrayTensor(data) {
  if(data !== undefined && !(data instanceof Array)) {
    data = [data];
  }
  var innerData = data;
  var shape = [];
  while(innerData instanceof Array) {
    shape.push(innerData.length);
    innerData = innerData[0];
  }
  return shape;
}

class Tensor {
  constructor(opts) {
    if(opts instanceof Array) {
      opts = {data: opts};
    }
    if(!isNaN(opts)) {
      opts = {data: [opts]};
    }
    var {shape, numDimensions, strides, initial_offset, data} = opts;
    if(data !== undefined) {
      if(shape === undefined)
        shape = parseArrayTensor(data);
      data = tensorUtil.flattenArray(data);
    }
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
        for(let i=numDimensions-1; i>=0; i--) {
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

    this.sparse = false;
    this.shape = shape;
    this.numDimensions = numDimensions;
    this.strides = strides;
    this.initial_offset = initial_offset;
    this.data = data;
  }

  at(coords) {
    if(!(coords instanceof Array))
      coords = [...arguments];
    var offset = this.initial_offset;
    for(let i=0; i<this.numDimensions; i++) {
      if(coords[i] >= this.shape[i] || coords[i]<0) {
        throw new Error('coordinate out of range! coord:' + coords[i] + ' dimension: ' + this.shape[i]);
      }
      offset += coords[i] * this.strides[i];
    }
    return this.data[offset];
  }

  set(coords, value) {
    if(!(coords instanceof Array))
      coords = [...arguments];
    var offset = this.initial_offset;
    for(let i=0; i<this.numDimensions; i++) {
      if(coords[i] >= this.shape[i] || coords[i]<0) {
        throw new Error('coordinate out of range! coord:' + coords[i] + ' dimension: ' + this.shape[i]);
      }
      offset += coords[i] * this.strides[i];
    }
    this.data[offset] = value;
    return value;
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

    tensorBinding.scale(this, compactified, 1);

    return compactified;
  }

  contract(otherTensor, dimsToContract, dest) {
    return contract(this, otherTensor, dimsToContract, dest);
  }

  outerProduct(otherTensor, dest) {
    return outerProduct(this, otherTensor, dest);
  }

  transpose() {
    return transpose(this);
  }

  fillNormal(mean, stdDev) {
    return fillNormal(mean, stdDev, this);
  }

  fillUniform(low, high) {
    return fillUniform(low, high, this);
  }

  sum() {
    return sum(this);
  }

  scale(x) {
    return scale(this, x);
  }

}
exports.Tensor = Tensor;

function transpose(tensor) {
  var shape = tensor.shape.slice(0).reverse();
  var strides = tensor.strides.slice(0).reverse();
  var initial_offset = tensor.initial_offset;
  var data = tensor.data;
  var numDimensions = tensor.numDimensions;
  var T = new Tensor({shape, strides, initial_offset, data, numDimensions});
  return T;
}
exports.transpose = transpose;

function fillNormal(mean, stdDev, dest) {
  tensorBinding.fillNormal(mean, stdDev, dest);
  return dest;
}
exports.fillNormal = fillNormal;

function fillUniform(low, high, dest) {
  tensorBinding.fillUniform(low, high, dest);
  return dest;
}
exports.fillUniform = fillUniform;

function print2DTensor(tensor) {
  var strings = [];
  for(let i=0; i<tensor.shape[0]; i++) {
    for(let j=0; j<tensor.shape[1]; j++) {
      strings.push(tensor.at([i,j]));
      strings.push(' ');
    }
    strings.push('\n');
  }
  return strings.join('');
}
exports.print2DTensor = print2DTensor;

function print1DTensor(tensor) {
  var strings = [];
  for(let i=0; i<tensor.shape[0]; i++) {
    strings.push(tensor.at([i]));
    strings.push(' ');
  }
  return strings.join('');
}
exports.print1DTensor = print1DTensor;

function printTensor(tensor) {
  if(tensor.numDimensions == 2)
    return print2DTensor(tensor);
  if(tensor.numDimensions == 1)
    return print1DTensor(tensor);
  return tensor;
}
exports.printTensor = printTensor;

function zerosLike(shape) {
  if(shape instanceof Tensor) {
    shape = shape.shape;
  }
  return new Tensor({shape});
}
exports.zerosLike = zerosLike;

function uniformLike(shape, low, high) {
  if(shape instanceof Tensor) {
    shape = shape.shape;
  }
  return (new Tensor({shape})).fillUniform(low, high);
}
exports.uniformLike = uniformLike;

function normalLike(shape, mean, stdDev) {
  if(shape instanceof Tensor) {
    shape = shape.shape;
  }
  return (new Tensor({shape})).fillNormal(mean, stdDev);
}
exports.normalLike = normalLike;

function onesLike(shape) {
  if(shape instanceof Tensor) {
    shape = shape.shape;
  }
  ones = new Tensor({shape});
  ones.data.fill(1.0);
  return ones;
}
exports.onesLike = onesLike;

function fillLike(value, shape) {
  if(shape instanceof Tensor) {
    shape = shape.shape;
  }
  ones = new Tensor({shape});
  ones.data.fill(value);
  return ones;
}
exports.fillLike = fillLike;

function broadcastShape(tensor1, tensor2) {
  var numDimensions = Math.max(tensor1.numDimensions, tensor2.numDimensions);
  var shape = new DimensionType(numDimensions);
  for(let i=0; i<numDimensions; i++) {
    var dimension1 = 1;
    var dimension2 = 1;
    if(i<tensor1.numDimensions)
      dimension1 = tensor1.shape[tensor1.numDimensions - i -1];

    if(i<tensor2.numDimensions)
      dimension2= tensor2.shape[tensor2.numDimensions - i -1];

    shape[numDimensions - i -1] = Math.max(dimension1, dimension2);
  }
  return shape;
}
exports.broadcastShape = broadcastShape;


function contract(source1, source2, dimsToContract, dest) {
  if(dest === undefined) {
    let shape = [];
    if(dimsToContract===0) {
      for(let dim of source1.shape) {
        shape.push(dim);
      }
      for(let dim of source2.shape) {
        shape.push(dim);
      }  
    } else {
      for(let dim of source1.shape.slice(0,-dimsToContract)) {
        shape.push(dim);
      }
      for(let dim of source2.shape.slice(dimsToContract)) {
        shape.push(dim);
      }
    }
    if(shape.length === 0)
      shape = [1];
  
    dest = new Tensor({shape});
  }
  tensorBinding.contract(source1, source2, dimsToContract,dest);
  return dest;
}
exports.contract = contract;

function outerProduct(source1, source2, dest) {
  return contract(source1, source2, 0, dest);
}

function numberToTensor(number) {
  if(number instanceof Tensor)
    return number;

  return new Tensor({shape:[1], data: new DataStorageType([number])});
}
exports.numberToTensor = numberToTensor;

function addScale(source1, source2, scale1, scale2, dest) {
  source1 = numberToTensor(source1);
  source2 = numberToTensor(source2);
  if(dest === undefined)
    dest = zerosLike(broadcastShape(source1, source2));
  dest = numberToTensor(dest);
  tensorBinding.addScale(source1, source2, scale1, scale2, dest);
  return dest;
}
exports.addScale = addScale;

function multiplyScale(source1, source2, scale, dest) {
  source1 = numberToTensor(source1);
  source2 = numberToTensor(source2);
  if(dest === undefined)
    dest = zerosLike(broadcastShape(source1, source2));
  dest = numberToTensor(dest);
  tensorBinding.multiplyScale(source1, source2, scale, dest);
  return dest;
}
exports.multiplyScale = multiplyScale;

function divideScale(source1, source2, scale, dest) {
  source1 = numberToTensor(source1);
  source2 = numberToTensor(source2);
  if(dest === undefined)
    dest = zerosLike(broadcastShape(source1, source2));
  dest = numberToTensor(dest);

  tensorBinding.divideScale(source1, source2, scale, dest);
  return dest;
}
exports.divideScale = divideScale;

function scale(source, scale, dest) {
  source = numberToTensor(source);
  if(dest === undefined)
    dest = zerosLike(source);
  tensorBinding.scale(source, scale, dest);
  return dest;
}
exports.scale = scale;

function sum(source) {
  return numberToTensor(tensorBinding.sum(source));
}
exports.sum = sum;

