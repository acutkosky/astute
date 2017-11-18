/* jshint esversion:6 */

var tensor = require('./src/tensor');
var autograd = require('./src/autograd');
var optim = require('./src/optim');


exports.tensor = tensor;
exports.optim = optim;
exports.autograd = autograd;