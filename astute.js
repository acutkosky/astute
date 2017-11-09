/* jshint esversion:6 */

var tensor = require('src/tensor.js');
var mathops = require('src/mathops.js');

for (let key in tensor) {
  exports[key] = tensor[key];
}

for (let key in mathops) {
  exports[key] = mathops[key];
}
