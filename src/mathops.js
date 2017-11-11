tensor = require('./tensor');
nodetensor = require('../build/Release/tensorBinding');

function exportOp(opname) {
  function op(source, dest) {
    source = tensor.numberToTensor(source);
    if(dest === undefined)
      dest = tensor.zerosLike(source);
    dest = tensor.numberToTensor(dest);

    nodetensor[opname](source, dest);

    return dest;
  }
  exports[opname] = op;
}


function exportBinaryOp(opname) {
  function binaryOp(source1, source2, dest) {
    source1 = tensor.numberToTensor(source1);
    source2 = tensor.numberToTensor(source2);
    if(dest === undefined)
      dest = tensor.zerosLike(tensor.broadcastShape(source1, source2));
    dest = tensor.numberToTensor(dest);

    nodetensor[opname](source1, source2, dest);

    return dest;
  }
  exports[opname] = binaryOp;
}

function add(source1, source2, dest) {
  return tensor.addScale(source1, source2, 1, 1, dest);
}
exports.add = add;

function sub(source1, source2, dest) {
  return tensor.addScale(source1, source2, 1, -1, dest);
}
exports.sub = sub;

function mul(source1, source2, dest) {
  return tensor.multiplyScale(source1, source2, 1, dest);
}
exports.mul = mul;

function div(source1, source2, dest) {
  return tensor.divideScale(source1, source2, 1, dest);
}
exports.div = div;

function dot(source1, source2, dest) {
  return tensor.matMul(source1, source2, dest);
}
exports.dot = dot;



exportOp('exp');
exportOp('abs');
exportOp('sqrt');
exportOp('sin');
exportOp('cos');
exportOp('tan');
exportOp('sinh');
exportOp('cosh');
exportOp('tanh');
exportOp('log');
exportOp('atan');
exportOp('acos');
exportOp('asin');
exportOp('atanh');
exportOp('acosh');
exportOp('asinh');
exportOp('erf');
exportOp('floor');
exportOp('ceil');
exportOp('round');
exportOp('sign');

exportBinaryOp('pow');
exportBinaryOp('fmod');
