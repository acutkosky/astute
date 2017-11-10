/* jshint esversion: 6 */

autograd = require('./autograd');
tensor = require('./tensor');
mathops = require('./mathops');

exports.utilityFuncs = [];

class Add extends autograd.Operation {
  forward(x, y) {
    if(!tensor.sameShape(x.data, y.data))
      throw "arguments must have same shape!";

    return tensor.addScale(x.data, y.data, 1, 1);
  }

  backward(outputDerivative) {
    return [outputDerivative, outputDerivative];
  }
}
exports.Add = Add;

function add(x, y) {
  return (new Add())(x,y);
}
exports.add = add;
exports.utilityFuncs.push(add);


class Sub extends autograd.Operation {
  forward(x, y) {
    if(!tensor.sameShape(x.data, y.data))
      throw "arguments must have same shape!";
    return tensor.addScale(x.data, y.data, 1, -1);
  }

  backward(outputDerivative) {
    return [outputDerivative, tensor.scale(outputDerivative, -1)];
  }
}
exports.Sub = Sub;

function sub(x, y) {
  return (new Sub())(x,y);
}
exports.sub = sub;
exports.utilityFuncs.push(sub);

class Mul extends autograd.Operation {
  forward(x, y) {
    if(!tensor.sameShape(x.data, y.data))
      throw "arguments must have same shape!";
    this.saveForBackward([y.data, x.data]);
    return tensor.multiplyScale(x.data, y.data, 1);
  }

  backward(outputDerivative) {
    var [ydata, xdata] = this.getSavedData();
    return [tensor.multiplyScale(outputDerivative, ydata, 1), tensor.multiplyScale(outputDerivative, xdata, 1)];
  }
}
exports.Mul = Mul;

function mul(x, y) {
  return (new Mul())(x,y);
}
exports.mul = mul;
exports.utilityFuncs.push(mul);

class Div extends autograd.Operation {
  forward(x, y) {
    if(!tensor.sameShape(x.data, y.data))
      throw "arguments must have same shape!";
    this.saveForBackward([y.data, x.data]);
    return tensor.divideScale(x.data, y.data, 1);
  }

  backward(outputDerivative) {
    var [ydata, xdata] = this.getSavedData();
    return [tensor.divideScale(outputDerivative, ydata, 1), tensor.divideScale(outputDerivative, xdata, 1)];
  }
}
exports.Div = Div;

function div(x, y) {
  return (new Div())(x,y);
}
exports.div = div;
exports.utilityFuncs.push(div);

class Scale extends autograd.Operation {
  constructor(x) {
    super();
    this.x = x;
  }
  forward(v) {
    return tensor.scale(v.data, this.x);
  }

  backward(outputDerivative) {
    return tensor.scale(outputDerivative, this.x);
  }
}
exports.Scale = Scale;

function scale(x, v) {
  return (new Scale(x))(v);
}
exports.scale = scale;
exports.utilityFuncs.push(scale);

class AddScalar extends autograd.Operation {
  constructor(x) {
    super();
    this.x = x;
  }
  forward(v) {
    return tensor.addScale(v.data, this.x, 1);
  }

  backward(outputDerivative) {
    return outputDerivative;
  }
}
exports.AddScalar = AddScalar;

function addScalar(x, v) {
  return (new AddScalar(x))(v);
}
exports.addScalar = addScalar;
exports.utilityFuncs.push(addScalar);

class Dot extends autograd.Operation {

  forward(x, y) {
    this.saveForBackward([x.data , y.data]);
    return tensor.matMul(x.data , y.data);
  }

  backward(outputDerivative) {
    var [xdata , ydata] = this.getSavedData();
    var Dx = null;
    var Dy = null;
    if(!this.parents[0].stopGrad)
      Dx = tensor.multiplyScale(ydata, outputDerivative, 1);
    if(!this.parents[1].stopGrad)
      Dy = tensor.multiplyScale(xdata, outputDerivative, 1);
    return [Dx, Dy];
  }
}
exports.Dot = Dot;

function dot(x, y) {
  return (new Dot())(x, y);
}
exports.dot = dot;
exports.utilityFuncs.push(dot);

class Square extends autograd.Operation {

  forward(x) {
    this.saveForBackward(x.data);
    return tensor.multiplyScale(x.data, x.data, 1);
  }

  backward(outputDerivative) {
    var inputData = this.getSavedData();
    return tensor.multiplyScale(outputDerivative, inputData, 2);
  }
}
exports.Square = Square;

function square(x) {
  return (new Square())(x);
}
exports.square = square;
exports.utilityFuncs.push(square);