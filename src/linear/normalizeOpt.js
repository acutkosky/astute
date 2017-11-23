/* jshint esversion: 6 */
var tensor = require('../tensor');
var autograd = require('../autograd');
var example = require('./example');

var EPSILON = 0.0001;

function  updateScalings(scalings, feature) {
  scalings.data.max(feature.data.abs(), scalings.data);
}

class NormalizedOptimizer {
  constructor(Optimizer, weights) {
    if(!isNaN(weights)) {
      weights = tensor.zerosLike([weights]);
    }
    weights = autograd.makeVariable(weights);
    this.optimizer = new Optimizer({vars: [weights]});
    this.scalings = new autograd.Variable(
      tensor.fillLike(weights.data, EPSILON),
      {
        requiresGrad: false,
        stopGrad: true
      });
    this.weights = weights;

    this.iterations = 0;
    this.cumulativeLoss = 0;
  }

  update(lossFunc, example) {
    updateScalings(this.scalings, example.feature);

    var loss = lossFunc(this.weights.dot(example.feature.div(this.scalings)), example.label);
    this.optimizer.step(loss);

    this.iterations++;
    this.cumulativeLoss += loss.data.data[0];
  }

  averageLoss() {
    if(this.iterations === 0) {
      return 0;
    }
    return this.cumulativeLoss/this.iterations;
  }

  getWeights() {
    return this.weights.data.div(this.scalings.data);
  }
}
exports.NormalizedOptimizer = NormalizedOptimizer;

function normalizeOptimizer(Optimizer, weights) {
  return new NormalizedOptimizer(Optimizer, weights);
}
exports.normalizeOptimizer = normalizeOptimizer;