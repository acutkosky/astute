tensor = require('./src/tensor.js');
nodetensor = require('./build/Release/tensorBinding');

function exportOp(opname) {
  function op(source, dest) {
    source1 = tensor.numberToTensor(source);
    if(dest === undefined)
      dest = tensor.zerosLike(source);

    nodetensor[opname](source, dest);

    return dest;
  }
  exports[opname] = binaryOp;
}


function exportBinaryOp(opname) {
  function binaryOp(source1, source2, dest) {
    source1 = tensor.numberToTensor(source1);
    source2 = tensor.numberToTensor(source2);
    if(dest === undefined)
      dest = tensor.zerosLike(tensor.broadcastShape(source1, source2));

    nodetensor[opname](source1, source2, dest);

    return dest;
  }
  tensor.prototype.
  exports[opname] = binaryOp;
}


exportOp(exp);
exportOp(abs);
exportOp(sqrt);
exportOp(sin);
exportOp(cos);
exportOp(tan);
exportOp(sinh);
exportOp(cosh);
exportOp(tanh);
exportOp(log);
exportOp(atan);
exportOp(acos);
exportOp(asin);
exportOp(atanh);
exportOp(acosh);
exportOp(asinh);
exportOp(erf);
exportOp(floor);
exportOp(ceil);
exportOp(round);

exportBinaryOp(exp);
exportBinaryOp(fmod);
