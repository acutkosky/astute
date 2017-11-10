/* jshint esversion:6 */

var tensor = require('./src/tensor');
var mathops = require('./src/mathops');
var autograd = require('./src/autograd');
var autogradOps = require('./src/autogradOps');

exports.tensor = tensor;
for (let key in mathops) {
  exports.tensor[key] = mathops[key];
}

exports.autograd = autograd;
for (let key in autogradOps) {
  if(key instanceof Function)
    exports.autograd[key] = autogradOps[key];
}

function firstArgisThis(func) {
  return function() {
    args = [this].concat([...arguments]);
    return func.apply(this, args);
  };
}

for (let i=0; i<autogradOps.utilityFuncs.length; i++) {
  let func = autogradOps.utilityFuncs[i];
  autograd.Variable.prototype[func.name] = firstArgisThis(func);
}