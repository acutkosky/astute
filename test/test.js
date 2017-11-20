/* jshint esversion: 6 */

var assert = require('assert');
var astute = require('../astute');

var {tensor, autograd, optim, linear} = astute;

describe('Tensor', function() {
  describe('constructor', function() {
    it('should construct empty tensor with given shape', function() {
      let T = new tensor.Tensor({shape: [2,3,4]});
      assert.deepEqual(T.data, new Float64Array(2*3*4));
      assert.deepEqual(T.shape, [2,3,4]);
      assert.deepEqual(T.strides, [12,4,1]);
    });

    it('should construct a tensor from a nested array', function() {
      let T = new tensor.Tensor(
        [[ [[999,2],[3,4]], [[5,6],[7,8]], [[9,   0],[1,2]], [[3,4],[5,6]] ],
         [ [[2  ,2],[9,4]], [[5,6],[7,8]], [[9,-888],[1,2]], [[3,4],[5,6]] ],
         [ [[4  ,2],[7,4]], [[5,6],[7,8]], [[9,   0],[1,2]], [[3,4],[5,6]] ]]);
      assert.deepEqual(T.shape, [3, 4, 2, 2]);
      assert.equal(T.at(0,0,0,0), 999);
      assert.equal(T.at(1,2,0,1), -888);
    });

    it('should construct a tensor from a number', function() {
      let T = new tensor.Tensor(4);
      assert.deepEqual(T.data, new Float64Array([4]));
    });

    it('should multiply a tensor by a number', function() {
      let T = new tensor.Tensor([1,2]);
      let scaled = tensor.scale(T, 3);//T.scale(3);
      assert.deepEqual(scaled.data, [3,6]);
    });

    it('should add tensors with broadcasting', function() {
      let T1 = new tensor.Tensor([[1,2],[3,4]]);
      let T2 = new tensor.Tensor([1,2]);
      let T3 = T1.add(T2);
      assert.deepEqual(T3.data, [2, 4, 4, 6]);
    });

  });

  describe('contract', function() {
    it('should multiply two matrices', function() {
      let T1 = new tensor.Tensor([[1,2],[3,4]]);
      let T2 = new tensor.Tensor([[2,3],[4,5]]);
      let T3 = T1.matMul(T2);

      assert.deepEqual(T3.data, [10,13,
                                 22,29]);
    });
    it('should multiply a matrix by a transposed matrix', function() {
      let T1 = new tensor.Tensor([[1,2,3],[3,4,5]]);
      let T2 = new tensor.Tensor([[1,2,0],[2,0,2]]);
      let T3 = T1.matMul(T2.transpose());

      assert.deepEqual(T3.data, [  5,  8,
                                  11, 16 ]);
    });
    it('should contract a higher order tensor', function() {
      let T1 = new tensor.Tensor([[[1,2],[3,4]],[[5,6],[7,8]]]);
      let T2 = new tensor.Tensor([1,2]);
      let T3 = T1.contract(T2, 1);

      assert.deepEqual(T3.shape, [2,2]);

      assert.deepEqual(T3.data, [5,11,17,23]);
    });
    it('should compute a dot product', function() {
      let T1 = tensor.onesLike([30]);
      var r = [];
      for(let i=0; i<30; i++) {
        r.push(i);
      }
      let T2 = new tensor.Tensor(r);
      let T3 = T1.dot(T2);
      assert.deepEqual(T3.data, [29*30/2]);
    });
    it('should multiply a matrix by a vector', function() {
      let T1 = new tensor.Tensor([[1,2,3],[4,5,6]]);
      let T2 = new tensor.Tensor([1,2,3]);
      let T3 = T1.matMul(T2);
      assert.deepEqual(T3.data, [14, 32]);
    });
    it('should compute an outer-product', function() {
      let T1 = new tensor.Tensor([1,2]);
      let T2 = new tensor.Tensor([2,3]);
      let T3 = T1.outerProduct(T2);
      assert.deepEqual(T3.data, [2,3,4,6]);
    });
  });

  describe('Sparse Vector', function() {
    it('should create sparse vectors', function() {
      var ST = new tensor.SparseVector([[1,123], [34, 23423]]);
      assert.equal(ST.at(34), 23423);
      assert.equal(ST.at([1]), 123);
      assert.equal(ST.at(3), 0);

      ST.set(7, -23);
      assert.equal(ST.at(7), -23);
    });

    it('should compute dot products with dense tensors', function() {
      var ST = new tensor.SparseVector([[0,4], [4,11]]);
      var T = new tensor.Tensor([1,2,3,4,5,6,7]);
      var dp = ST.dot(T);

      assert.equal(dp, 59);

      dp = T.dot(ST);
      assert.equal(dp, 59);
    });

    it('should compute dot products with sparse tensors', function() {
      var ST1 = new tensor.SparseVector([[0,4], [4,11], [9, 3]]);
      var ST2 = new tensor.SparseVector([[23, 343], [9,2], [0,5]]);
      var dp = ST1.dot(ST2);

      assert.equal(dp, 26);
    });

    it('should multiply sparse vectors by dense matrices', function() {
      var M1 = new tensor.Tensor([[1,2],[3,4],[5,6]]);
      var S1 = new tensor.SparseVector([[0,1],[2,4]], 3);
      var p1 = S1.matMul(M1);
      assert.deepEqual(p1.data, [21, 26]);

      var M2 = new tensor.Tensor([[1,2,3],[3,4,5]]);
      var S2 = new tensor.SparseVector([[0,2]], 2);
      var p2 = M2.matMul(S2);
      assert.deepEqual(p2.data, [2, 6]);
    });

    it('should add sparse vectors', function() {
      let S1 = new tensor.SparseVector([[0,2]]);
      let S2 = new tensor.SparseVector([[0,4], [1,3]]);
      let S3 = S1.add(S2);

      assert.equal(S3.at(0), 6);
      assert.equal(S3.at(1), 3);
      assert.equal(S3.at(2), 0);
    });
    it('should add sparse vectors to dense vectors', function() {
      let S1 = new tensor.SparseVector([[0,2]], 3);
      let T2 = new tensor.Tensor([4,3,0]);
      let S3 = S1.add(T2);
      assert.equal(S3.at(0), 6);
      assert.equal(S3.at(1), 3);
      assert.equal(S3.at(2), 0);
    });
  });

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

  describe('linear', function() {
    it('should compute correct logistic regression loss', function() {
      var weights = new tensor.Tensor({shape: [5]});
      weights.set(0,3);
      var feature = new tensor.SparseVector([[0,2],[3,-5]], 5);
      var label = -1.0;
      var example = new linear.Example(feature, label);

      var loss = linear.loss.logisticLoss(example.feature.dot(weights), label);

      assertSmall(loss.data.data[0] - 6.0024756851377301);
    });
  });
});
