/* jshint esversion:6 */

var tensor = require('./src/tensor');
var mathops = require('./src/mathops');

for (let key in tensor) {
  exports[key] = tensor[key];
}

for (let key in mathops) {
  exports[key] = mathops[key];
}
