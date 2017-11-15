/* jshint esversion:6 */


optim = require('./optimizer');
tensor = require('../tensor');

var EPSILON = 0.000001;

class AdaGrad extends optim.Optimizer {
  constructor(opts) {
    super(opts);
    this.lr = opts.lr;
  }

  makeSlots(vars) {
    for(let v of vars) {
      this.setSlot(v, 'sumGradSq', tensor.fillLike(v.data, EPSILON));
    }
  }

  applyGrads(vars) {
    this.t += 1;
    for(let v of vars) {
      var sumGradSq = this.getSlot(v, 'sumGradSq');
      sumGradSq = tensor.addScale(v.grad.square(), sumGradSq, 1, 1, sumGradSq);
      v.data = v.data.add(v.grad.divideScale(sumGradSq.sqrt(), -this.lr));
      this.setSlot(v, 'sumGradSq', sumGradSq);
    }
  }
}
module.exports = AdaGrad;
