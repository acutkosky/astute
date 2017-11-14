/* jshint esversion: 6 */

var optimizer = require('./optimizer');
var SGD = require('./SGD');
var AdaGrad = require('./adagrad');


module.exports = {
  optimizer,
  SGD,
  AdaGrad
};