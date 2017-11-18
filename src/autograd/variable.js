/* jshint esversion: 6 */ 
var tensor = require('../tensor');

class Variable {
  constructor(data, opts) {
    opts = opts || {};
    var {stopGrad, requiresGrad} = opts;
    if(stopGrad === undefined)
      stopGrad = false;
    if(requiresGrad === undefined)
      requiresGrad = true;

    if(!(data instanceof tensor.Tensor) && !(data instanceof tensor.SparseVector))
      data = new tensor.Tensor(data);
    this.data = data;
    this.grad = undefined;
    this.parent = undefined;
    this.stopGrad = stopGrad;
    this.requiresGrad = requiresGrad;
    this.children = [];
  }

  backward(derivative) {
    if(derivative === undefined) {
      derivative = 1.0;
    }
    if(this.grad === undefined) {
      this.grad = derivative;
    } else {
      tensor.addScale(this.grad, derivative, 1, this.grad);
    }

    if(this.parent !== undefined && !this.stopGrad)
      this.parent.backwardWrapper(derivative);
  }

  zeroGrad() {
    this.grad = undefined;
    if(this.parent !== undefined)
      this.parent.zeroGrad();
  }
}
exports.Variable = Variable;



//base class for operations
class Operation {

  constructor() {
    var applyOp = function() {
      return this.forwardWrapper.apply(this, arguments);
    }.bind(this);

    this.savedData = {};
    this.child = null;
    this.parents = [];

    for(let key in this) {
      applyOp[key] = this[key];
    }

    return applyOp;
  }

  saveForBackward(info) {
    this.savedData = info;
  }

  getSavedData() {
    return this.savedData;
  }

  forwardWrapper() {
    for(let i=0; i<arguments.length; i++) {
      let arg = arguments[i];
      if(!(arg instanceof Variable)) {
        arg = new Variable(arg);
        arguments[i] = arg;
      }
      arg.children.push(this);
      this.parents.push(arg);
    }
    var output = this.forward(...arguments);
    if(!(output instanceof Variable))
      output = new Variable(output);
    output.parent = this;
    this.child = output;
    return output;
  }

  backwardWrapper(outputDerivative) {
    var inputDerivatives = {};
    for(let i=0; i<this.parents.length; i++) {
      if(this.parents[i].requiresGrad) {
        inputDerivatives[i] = this.backward(outputDerivative, i);
      }
    }

    for(let i = 0; i<this.parents.length; i++) {
      if(this.parents[i].requiresGrad) {
        this.parents[i].backward(inputDerivatives[i]);
      }
    }
  }

  zeroGrad() {
    for(let i = 0; i<this.parents.length; i++)
      this.parents[i].zeroGrad();
  }

  /**
    * should take some number of arguments of type Variable.
    * should return exactly one Variable.
    * TODO: think about operations with fan-out
    */
  forward() {
    throw new Error('Forward Not Implemented!');
  }

  /**
    * outputDerivative: a tensor argument representing
    * the derivative of the final loss with respect
    * to the output of this operation.
    * argIndex: an integer between 0 and N-1 where N is th number
    * of input arguments to this.forward. argIndex specifies 
    * which input variable we should return the derivative for.
    * 
    * Should return a tensor corresponding to the derivative of
    * the loss with respect to the argIndex'th input to this.forward.
    */
  backward(outputDerivative, argIndex) {
    throw new Error('Backward Not Implemented!');
  }
}
exports.Operation = Operation;


