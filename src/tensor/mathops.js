/* jshint esversion:6 */

var denseTensor = require('./denseTensor');
var sparseTensor = require('./sparseTensor');
var nodetensor = require('../../build/Release/tensorBinding');

var mathjs = require('mathjs');

function exportOp(opname, canSparse, opfunc) {
  if(opfunc === undefined)
    opfunc = Math[opname];
  // if(canSparse === undefined)
  //   canSparse = true;

  function op(source, dest) {
    if(source.sparse && !canSparse) {
      source = source.toDense();
    }
    if(source.sparse) {
      return source.apply(opfunc, dest);
    } else {
      source = denseTensor.numberToTensor(source);
      if(dest === undefined)
        dest = denseTensor.zerosLike(source);
      dest = denseTensor.numberToTensor(dest);

      nodetensor[opname](source, dest);

      return dest;
    }
  }
  exports[opname] = op;
}

function exportBinaryOp(opname, canSparse, commutative, opfunc) {
  if(opfunc === undefined)
    opfunc = Math[opname];
  if(canSparse === undefined)
    canSparse = true;

  function binaryOp(source1, source2, dest) {
    if(!isNaN(source2)) {
      source2 = denseTensor.numberToTensor(source2);
    }
    if(source1.sparse && !canSparse) {
      source1 = source1.toDense();
    }
    if(source2.sparse && !canSparse) {
      source2 = source2.toDense();
    }
    if(commutative) {
      if(source2.sparse && !source1.sparse) {
        let temp = source2;
        source2 = source1;
        source1 = temp;
      }
    }
    if(source1.sparse) {
      return source1.applyBinary(opfunc, source2, dest);
    } else {
      source1 = denseTensor.numberToTensor(source1);
      if(dest === undefined)
        dest = denseTensor.zerosLike(denseTensor.broadcastShape(source1, source2));
      dest = denseTensor.numberToTensor(dest);

      nodetensor[opname](source1, source2, dest);

      return dest;
    }
  }
  exports[opname] = binaryOp;
}

function add(source1, source2, dest) {
  return addScale(source1, source2, 1, 1, dest);
}
exports.add = add;

function sub(source1, source2, dest) {
  return addScale(source1, source2, 1, -1, dest);
}
exports.sub = sub;

function mul(source1, source2, dest) {
  return multiplyScale(source1, source2, 1, dest);
}
exports.mul = mul;

function div(source1, source2, dest) {
  return divideScale(source1, source2, 1, dest);
}
exports.div = div;

function multiplyScale(source1, source2, scale, dest) {
  if(source1.sparse) {
    return sparseTensor.multiplyScale(source1, source2, scale, dest);
  } else if(source2.sparse) {
    return sparseTensor.multiplyScale(source2, source1, scale, dest);
  } else {
    return denseTensor.multiplyScale(source1, source2, scale, dest);
  }  
}
exports.multiplyScale = multiplyScale;

function divideScale(source1, source2, scale, dest) {
  if(source1.sparse) {
    return sparseTensor.divideScale(source1, source2, scale, dest);
  } else if (source2.sparse) {
    throw new Error('Cannot divide by a sparse vector!');
  } else {
    return denseTensor.divideScale(source1, source2, scale, dest);
  }  
}
exports.divideScale = divideScale;

function addScale(source1, source2, scale1, scale2, dest) {
  if(source1.sparse) {
    return sparseTensor.addScale(source1, source2, scale1, scale2, dest);
  } else if(source2.sparse) {
    return sparseTensor.addScale(source2, source1, scale2, scale1, dest);
  } else {
    return denseTensor.addScale(source1, source2, scale1, scale2, dest);
  }
}
exports.addScale = addScale;

function scale(source, scale, dest) {
  if(source.sparse) {
    return sparseTensor.scale(source, scale, dest);
  } else {
    return denseTensor.scale(source, scale, dest);
  }
}
exports.scale = scale;

function sum(source) {
  if(source.sparse) {
    return sparseTensor.sum(source);
  } else {
    return denseTensor.sum(source);
  }
}
exports.sum = sum;

function dot(source1, source2, dest) {
  if(source2.sparse) {
    let dp = sparseTensor.dot(source2, source1, dest);
    // if(dest !== undefined)
    //   dest.set(0, dp);
    return dp;
  }
  if(source1.sparse) {
    let dp = sparseTensor.dot(source1, source2, dest);
    // if(dest !== undefined)
    //   dest.set(0, dp);
    return dp;
  }
  return matMul(source1, source2, dest);
}
exports.dot = dot;

function matMul(source1, source2, dest) {
  if(source1.sparse) {
    return sparseTensor.matMul(source1, source2, dest);
  } else if(source2.sparse) {
    return sparseTensor.matMul(source2, source1.transpose(), dest);
  } else {
    return denseTensor.contract(source1, source2, 1, dest);
  }
}
exports.matMul = matMul;

function sameShape(tensor1, tensor2) {
  if(tensor1.numDimensions != tensor2.numDimensions)
    return false;
  for(let i=0; i<tensor1.numDimensions; i++) {
    if(tensor1.shape[i] != tensor2.shape[i])
      return false;
  }
  return true;
}
exports.sameShape = sameShape;

function square(source, dest) {
  return multiplyScale(source, source, 1, dest);
}
exports.square = square;

exportOp('exp');
exportOp('abs', true);
exportOp('sqrt', true);
exportOp('sin', true);
exportOp('cos');
exportOp('tan', true);
exportOp('sinh');
exportOp('cosh');
exportOp('tanh');
exportOp('log', false);
exportOp('atan', true);
exportOp('acos');
exportOp('asin', true);
exportOp('atanh');
exportOp('acosh');
exportOp('asinh');
exportOp('erf', true, mathjs.erf);
exportOp('floor', true);
exportOp('ceil', true);
exportOp('round', true);
exportOp('sign', true);

exportBinaryOp('max', true, true);
exportBinaryOp('min', true, true);
exportBinaryOp('pow');
exportBinaryOp('fmod', (x, y) => {return x - Math.trunc(x/y) * y;});

