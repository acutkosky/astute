/* jshint esversion: 6 */

var assert = require('assert');
var astute = require('../astute');

var {tensor, autograd, optim, linear} = astute;

function assertSmall(value) {
  var epsilon = 0.0001;
  if(!isNaN(value)) {
    if(Math.abs(value) < epsilon)
      return true;
    else
      throw new Error(value + ' is not small!');
  }
  if(value instanceof tensor.Tensor) {
    if(value.numDimensions > 1 || value.shape[0]!=1){
      throw "Tensor arguments to isSmall must be scalars!";
    }
    if(Math.abs(value.data[0]) < epsilon)
      return true;
    else
      throw new Error(value.data[0] + ' is not small!');
  }
  throw new Error("Must supply a number or a Tensor to isSmall, supplied " +value);
}

describe('Linear Learning', function() {
  it('should compute correct logistic regression loss', function() {
    var weights = new tensor.Tensor({shape: [5]});
    weights.set(0,3);
    var feature = new tensor.SparseVector([[0,2],[3,-5]], 5);
    var label = -1.0;
    var example = new linear.Example(feature, label);

    var loss = linear.loss.logisticLoss(example.feature.dot(weights), label);

    assertSmall(loss.data.data[0] - 6.0024756851377301);
  });

  it('should optimize logistic regression', function() {
    var weights = new tensor.Tensor({shape: [5]});

    function getExample() {
      var feature = new tensor.SparseVector([], 5);
      var a = Math.random()*2-1;
      var b = Math.random()*2-1;
      feature.set(0, a);
      feature.set(3, b);
      var label = Math.sign(a-b);

      return new linear.Example(feature, label);
    }

    var normalizedOpt = linear.normalize.normalizeOptimizer(optim.FreeRex, weights);

    for(let t=0; t<1000; t++) {
      let example = getExample();
      normalizedOpt.update(linear.loss.logisticLoss, example);
    }
    weights = normalizedOpt.getWeights();
    var totalLoss = 0;
    for(let t=0; t<1000; t++) {
      let example = getExample();
      let loss = linear.loss.logisticLoss(weights.dot(example.feature.data), example.label);
      totalLoss += loss.data[0];
    }
    assert(totalLoss/1000 < 0.16);

  });
});