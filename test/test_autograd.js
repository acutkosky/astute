/* jshint esversion: 6 */

var assert = require('assert');
var astute = require('../astute');

var {tensor, autograd, optim, linear} = astute;

describe('AutoGrad', function() {

  function numericalGrad(opFunc, inputShapes, low, high, sparseInput) {
    var baseVariables = [];
    var baseTensors = [];
    var perturbedTensors = [];
    var noiseTensors = [];

    var range = high-low;
    var epsilon = range*0.000000001;

    for(let i=0 ;i<inputShapes.length; i++) {
      var T = undefined;
      if(sparseInput) {
        var uniformData = Math.random() * (high-low) + low;
        var randomIndex = Math.floor(Math.random() * inputShapes[0][0]);

        T = new tensor.SparseVector([[randomIndex, uniformData]], inputShapes[0][0]);
      } else {
        T = tensor.zerosLike(inputShapes[i]);
        T.fillUniform(low, high);
      }
      baseTensors.push(T);
      baseVariables.push(new autograd.Variable(T));

      var noise = tensor.random.normalLike(inputShapes[i], 0 , epsilon);
      var scaledEpsilon = epsilon/Math.sqrt(T.totalSize());
      noiseTensors.push(noise);
      perturbedTensors.push(tensor.mathops.addScale(noise, T, 1, 1));

    }
    var result = opFunc(...baseVariables).sum();
    var perturbedResult = opFunc(...perturbedTensors).sum();
    result.zeroGrad();
    result.backward();
    var diff = tensor.mathops.addScale(perturbedResult.data, result.data, 1, -1);

    var norm = new tensor.Tensor([0]);
    var directionalGrad = new tensor.Tensor([0]);
    for(let i=0; i<baseVariables.length; i++) {
      var currentNormSq = tensor.mathops.multiplyScale(noiseTensors[i], noiseTensors[i], 1).sum();
      tensor.mathops.addScale(norm, currentNormSq, 1, 1, norm);
      norm = norm.sqrt();
      var currentDotProduct = tensor.mathops.multiplyScale(noiseTensors[i], baseVariables[i].grad, 1).sum();
      tensor.mathops.addScale(directionalGrad, currentDotProduct, 1, 1, directionalGrad);
    }
    tensor.mathops.divideScale(directionalGrad, norm, 1, directionalGrad);
    var numDirectionalGrad = tensor.mathops.divideScale(diff, norm, 1);
    var error = tensor.mathops.addScale(numDirectionalGrad, directionalGrad, 1, -1);
    var dynamicRange = tensor.mathops.addScale(numDirectionalGrad.abs(), directionalGrad.abs(), 1, 1);
    var relativeError = tensor.mathops.divideScale(error, dynamicRange, 1);
    if(relativeError.sparse)
      relativeError = relativeError.toDense();
    return relativeError.data;
  }

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

  function testFunction(funcname, shapes, sparse) {
    var sparseString = '';
    if(sparse) {
      sparseString = 'sparse ';
    }
    it('differentiates '+sparseString+funcname, function() {
      for(let trial=0; trial<10; trial++) {
        var error = numericalGrad(autograd[funcname], shapes, 1, 10, sparse);
        assertSmall(error);
      }
    });
  }

  describe('autograd', function() {

    var unaryFuncs = [
    'square',
    'exp',
    'sqrt',
    'sin',
    'cos',
    'tan',
    'sum',
    'log'
    ];
    for(let i=0; i<unaryFuncs.length; i++) {
      testFunction(unaryFuncs[i], [[2,2,2]]);
    }

    for(let i=0; i<unaryFuncs.length; i++) {
      if(unaryFuncs[i] != 'log')
        testFunction(unaryFuncs[i], [[3]], true);
    }

    var binaryFuncs = [
    'dot',
    'add',
    'sub',
    'mul',
    'div'
    ];
    for(let i=0; i<binaryFuncs.length; i++) {
      testFunction(binaryFuncs[i], [[4], [4]]);
    }

    for(let i=0; i<binaryFuncs.length; i++) {
      if(binaryFuncs[i] != 'div')
        testFunction(binaryFuncs[i], [[1], [1]], true);
    }

    it('returns sparse gradient for dot product with sparse vector', function() {
      var S1 = new tensor.SparseVector([[0,3],[5,10]], 6);
      var T2 = new tensor.onesLike([6]);

      var V1 = new autograd.Variable(S1, {requiresGrad: false});
      var V2 = new autograd.Variable(T2);

      var dotp = V1.dot(V2);

      dotp.zeroGrad();
      dotp.backward();

      assert.equal(V1.grad, undefined);
      assert.equal(V2.grad.at(5), 10);
    });
  });

  describe('optimizer', function() {

    function testOptimizer(optimizerFactory) {
      var x = new autograd.Variable([10, 8]);
      var y = new autograd.Variable([3, -4], {requiresGrad: false});
      var opt = optimizerFactory([x]);

      for(let t=1; t<100; t++) {
        //loss = ( <x,y> - 6 )^2
        loss = x.dot(y).sub(6).square();
        opt.step(loss);
      }
      loss = x.dot(y).sub(6).square();
      assertSmall(loss.data);
    }

    it('SGD should optimize', function() {
      testOptimizer(vars => {return new optim.SGD({lr: 0.1, vars: vars});});
    });
    it('AdaGrad should optimize', function() {
      testOptimizer(vars => {return new optim.AdaGrad({lr: 1.0, vars: vars});});
    });
    it('FreeRex should optimize', function() {
      testOptimizer(vars => {return new optim.FreeRex({vars: vars});});
    });
  });
});
