/* jshint esversion: 6 */


//base class for optimizers
class Optimizer {
  constructor(vars) {
    this.vars = vars;
    this.slots = new Map();
    this.makeSlots(vars);
  }

  step(loss, vars) {
    loss.zeroGrad();
    loss.backward();
    if(vars === undefined)
      vars = this.vars;
    this.applyGrads(vars);
  }

  setSlot(v, name, value) {
    if(this.slots.get(v) === undefined)
      this.slots.set(v, new Map());
    this.slots.get(v).set(name, value);
  }

  makeSlots(vars) {
  }

  applyGrads(vars) {
    throw new Error('applyGrads not implemented!');
  }

  getSlot(v, name) {
    return this.slots.get(v).get(name);
  }
}
exports.Optimizer = Optimizer;