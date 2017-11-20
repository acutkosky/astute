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

      assert.equal(dp.data[0], 59);

      dp = T.dot(ST);
      assert.equal(dp.data[0], 59);
    });

    it('should compute dot products with sparse tensors', function() {
      var ST1 = new tensor.SparseVector([[0,4], [4,11], [9, 3]]);
      var ST2 = new tensor.SparseVector([[23, 343], [9,2], [0,5]]);
      var dp = ST1.dot(ST2);

      assert.equal(dp.data[0], 26);
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
});