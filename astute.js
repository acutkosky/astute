/* jshint esversion:6 */

var tensor = require('./src/tensor');
var mathops = require('./src/mathops');
var autograd = require('./src/autograd');
var autogradOps = require('./src/autogradOps');
var sparseTensor = require('./src/sparseTensor');

function firstArgisThis(func) {
  return function() {
    args = [this].concat([...arguments]);
    return func.apply(this, args);
  };
}

exports.tensor = tensor;
tensor.mathOps = {};
for (let key in mathops) {
  exports.tensor.mathOps[key] = mathops[key];
  if(mathops[key] instanceof Function) {
    let func = mathops[key];
    tensor.Tensor.prototype[key] = firstArgisThis(func);
    sparseTensor.SparseVector.prototype[key] = firstArgisThis(func);
  }
}

exports.autograd = autograd;
for (let key in autogradOps) {
  if(autogradOps[key] instanceof Function) {
    exports.autograd[key] = autogradOps[key];
  }

}

for (let i=0; i<autogradOps.utilityFuncs.length; i++) {
  let func = autogradOps.utilityFuncs[i];
  autograd.Variable.prototype[func.name] = firstArgisThis(func);
}

exports.sparseTensor = sparseTensor;
