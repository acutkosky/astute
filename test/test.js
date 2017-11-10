/* jshint esversion: 6 */

var assert = require('assert');
var astute = require('../astute');


describe('Tensor', function() {
  describe('constructor', function() {
    it('should construct empty tensor with given shape', function() {
      let T = new astute.tensor.Tensor({shape: [2,3,4]});
      assert.deepEqual(T.data, new Float64Array(2*3*4));
      assert.deepEqual(T.shape, [2,3,4]);
      assert.deepEqual(T.strides, [12,4,1]);
    });

    it('should construct a tensor from a nested array', function() {
      let T = new astute.tensor.Tensor(
        [[ [[999,2],[3,4]], [[5,6],[7,8]], [[9,   0],[1,2]], [[3,4],[5,6]] ],
         [ [[2  ,2],[9,4]], [[5,6],[7,8]], [[9,-888],[1,2]], [[3,4],[5,6]] ],
         [ [[4  ,2],[7,4]], [[5,6],[7,8]], [[9,   0],[1,2]], [[3,4],[5,6]] ]]);
      assert.deepEqual(T.shape, [3, 4, 2, 2]);
      assert.equal(T.at(0,0,0,0), 999);
      assert.equal(T.at(1,2,0,1), -888);
    });

    it('should construct a tensor from a number', function() {
      let T = new astute.tensor.Tensor(4);
      assert.deepEqual(T.data, new Float64Array([4]));
    });
  });

  describe('contract', function() {
    it('should correctly multiply two matrices', function() {
      let T1 = new astute.tensor.Tensor([[1,2],[3,4]]);
      let T2 = new astute.tensor.Tensor([[2,3],[4,5]]);
      let T3 = T1.matMul(T2);

      assert.deepEqual(T3.data, [10,13,
                                 22,29]);
    });
    it('should correctly multiply a matrix by a transposed matrix', function() {
      let T1 = new astute.tensor.Tensor([[1,2,3],[3,4,5]]);
      let T2 = new astute.tensor.Tensor([[1,2,0],[2,0,2]]);
      let T3 = T1.matMul(T2.transpose());

      assert.deepEqual(T3.data, [  5,  8,
                                  11, 16 ]);
    });
    it('should correctly contract a higher order tensor', function() {
      let T1 = new astute.tensor.Tensor([[[1,2],[3,4]],[[5,6],[7,8]]]);
      let T2 = new astute.tensor.Tensor([1,2]);
      let T3 = T1.contract(T2, 1);

      assert.deepEqual(T3.shape, [2,2]);

      assert.deepEqual(T3.data, [5,11,17,23]);
    });
    it('should correctly compute a dot product', function() {
      let T1 = new astute.tensor.onesLike([30]);
      var r = [];
      for(let i=0; i<30; i++) {
        r.push(i);
      }
      let T2 = new astute.tensor.Tensor(r);
      let T3 = T1.matMul(T2);
      assert.deepEqual(T3.data, [29*30/2]);
    });
  });
});
