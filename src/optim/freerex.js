/* jshint esversion:6 */


optim = require('./optimizer');
tensor = require('../tensor');

var EPSILON = 0.000001;

class FreeRex extends optim.Optimizer {
  constructor(opts) {
    super(opts);
    this.lr = opts.lr || 0.45;
  }

  makeSlots(vars) {
    for(let v of vars) {
      this.setSlot(v, 'sumGrad', tensor.zerosLike(v.data));
      this.setSlot(v, 'oneOverEtaSq', tensor.fillLike(v.data, EPSILON));
      this.setSlot(v, 'Lmax', tensor.zerosLike(v.data));
    }
  }

  applyGrads(vars) {
    this.t += 1;
    for(let v of vars) {
      var sumGrad = this.getSlot(v, 'sumGrad');
      var oneOverEtaSq = this.getSlot(v, 'oneOverEtaSq');
      var Lmax = this.getSlot(v, 'Lmax');

      var absGrad = v.grad.abs();
      var gradSquared = absGrad.square();
      var absSumGrad = sumGrad.abs();

      tensor.add(sumGrad, v.grad, sumGrad);
      tensor.max(Lmax, absGrad, Lmax);

      tensor.max(oneOverEtaSq.add(gradSquared.scale(2)), Lmax.mul(absSumGrad), oneOverEtaSq);

      var direction = sumGrad.sign();
      tensor.scale(direction, -1, direction);

      direction.mul( tensor.exp(absSumGrad.divideScale(oneOverEtaSq.sqrt(), this.lr)).sub(1) , v.data);

      this.setSlot(v, 'sumGrad', sumGrad);
      this.setSlot(v, 'oneOverEtaSq', oneOverEtaSq);
      this.setSlot(v, 'Lmax', Lmax);
    }
  }
}
module.exports = FreeRex;
