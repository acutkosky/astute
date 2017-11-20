/* jshint esversion: 6 */
var tensor = require('../tensor');
var autograd = require('../autograd');


class Example {
  constructor(feature, label, weight=1.0) {
    this.feature = autograd.makeVariable(
      feature, 
      {stopGrad: true, requiresGrad: false}
      );
    this.weight = weight;
    this.label = label;
  }
}
module.exports = Example;

