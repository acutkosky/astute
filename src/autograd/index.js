/* jshint esversion: 6 */

var autogradOps = require('./autogradOps');
var variable = require('./variable');

function firstArgisThis(func) {
  return function() {
    args = [this].concat([...arguments]);
    return func.apply(this, args);
  };
}

for(let key in autogradOps) {
  if(autogradOps[key] instanceof Function) {
    exports[key] = autogradOps[key];
  }
}

Object.assign(exports, variable);
for(let key in variable) {
  exports[key] = variable[key];
}


for (let i=0; i<autogradOps.utilityFuncs.length; i++) {
  let func = autogradOps.utilityFuncs[i];
  exports.Variable.prototype[func.name] = firstArgisThis(func);
}

function makeVariable(toMake, opts) {
  if(!(toMake instanceof variable.Variable)) {
    return new variable.Variable(toMake, opts);
  } else {
    return Object.assign(toMake, opts);
  }
}
exports.makeVariable = makeVariable;