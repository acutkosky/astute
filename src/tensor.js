/* jshint esversion:6 */

var denseTensor = require('./denseTensor');
var sparseTensor = require('./sparseTensor');
var mathops = require('./mathops');


exports.denseTensor = denseTensor;
exports.mathops = mathops;
exports.sparseTensor = sparseTensor;

exports.Tensor = denseTensor.Tensor;
exports.SparseVector = sparseTensor.SparseVector;

function firstArgisThis(func) {
  return function() {
    args = [this].concat([...arguments]);
    return func.apply(this, args);
  };
}


for (let key in mathops) {
  exports[key] = mathops[key];
  if(mathops[key] instanceof Function) {
    let func = mathops[key];
    exports.Tensor.prototype[key] = firstArgisThis(func);
    exports.SparseVector.prototype[key] = firstArgisThis(func);
  }
}


//surface commonly-used functions
exports.onesLike = denseTensor.onesLike;
exports.zerosLike = denseTensor.zerosLike;
exports.fillLIke = denseTensor.fillLike;

exports.random = {};
exports.random.uniformLike = denseTensor.uniformLike;
exports.random.normalLike = denseTensor.normalLike;
