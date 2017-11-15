/* jshint esversion: 6 */

var optimizer = require('./optimizer');
var SGD = require('./SGD');
var AdaGrad = require('./adagrad');
var FreeRex = require('./freerex');


module.exports = {
  optimizer,
  SGD,
  AdaGrad,
  FreeRex
};