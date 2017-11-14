/* jshint esversion:6 */

var tensor = require('./src/tensor');
var autograd = require('./src/autograd');
var autogradOps = require('./src/autogradOps');


exports.tensor = tensor;


function firstArgisThis(func) {
  return function() {
    args = [this].concat([...arguments]);
    return func.apply(this, args);
  };
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
