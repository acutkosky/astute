/* jshint esversion: 6 */ 
var tensor = require('./tensor');


class Variable {
  constructor(data, stopGrad=false) {
    if(!(data instanceof tensor.Tensor))
      data = new tensor.Tensor(data);
    this.data = data;
    this.grad = null;
    this.parent = null;
    this.stopGrad = stopGrad;
    this.children = [];
  }

  backward(derivative) {
    // console.log(derivative);
    // if(derivative!== null)
    //   console.log("back derivative: ",derivative.data);
    if(this.grad === null) {
      this.grad = derivative;
    } else {
      tensor.addScale(this.grad, derivative, 1, this.grad);
    }

    if(this.parent !== null && !this.stopGrad)
      this.parent.backwardWrapper(derivative);
  }

  zeroGrad() {
    this.grad = null;
    if(this.parent !== null)
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
    var output = this.forward.apply(this, arguments);
    if(!(output instanceof Variable))
      output = new Variable(output);
    output.parent = this;
    this.child = output;
    return output;
  }

  backwardWrapper(outputDerivative) {
    var inputDerivatives = this.backward(outputDerivative);
    if(!(inputDerivatives instanceof Array) && this.parents.length == 1) {
      inputDerivatives = [inputDerivatives];
    }
    for(let i = 0; i<this.parents.length; i++)
      this.parents[i].backward(inputDerivatives[i]);
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
    throw 'Forward Not Implemented!';
  }

  /**
    * takes a single tensor argument representing
    * the derivative of the final output with respect
    * to the output of this operation.
    * should return an array of tensors corresponding to the
    * arguments to this.forward. The ith element of this array
    * is the derivative of the final output with respect to the
    * ith input to this.forward.
    */
  backward(outputDerivative) {
    throw 'Backward Not Implemented!';
  }
}
exports.Operation = Operation;
