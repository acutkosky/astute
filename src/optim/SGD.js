/* jshint esversion:6 */


optim = require('./optimizer');

class SGD extends optim.Optimizer {
  constructor(lr, vars) {
    super(vars);
    this.lr = lr;
    this.t = 0;
  }

  applyGrads(vars) {
    this.t += 1;
    for(let v of vars) {
      v.data = v.data.add(v.grad.scale(-this.lr/Math.sqrt(this.t)));
    }
  }
}
module.exports = SGD;
