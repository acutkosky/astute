/* jshint esversion: 6 */

var autograd = require('./autograd');
var tensor = require('./tensor');
var mathops = tensor.mathops;
// require('./mathops');

exports.utilityFuncs = [];

class Add extends autograd.Operation {
  forward(x, y) {
    if(!mathops.sameShape(x.data, y.data))
      throw new Error("arguments must have same shape!");

    return mathops.addScale(x.data, y.data, 1, 1);
  }

  backward(outputDerivative, argIndex) {
    return outputDerivative;
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
    if(!mathops.sameShape(x.data, y.data))
      throw new Error("arguments must have same shape!");
    var answer = mathops.addScale(x.data, y.data, 1, -1);
    return mathops.addScale(x.data, y.data, 1, -1);
  }

  backward(outputDerivative, argIndex) {
    switch(argIndex) {
      case 0:
        return outputDerivative;
      case 1:
        var answer = mathops.scale(outputDerivative, -1);
        return mathops.scale(outputDerivative, -1);
    }
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
    if(!mathops.sameShape(x.data, y.data))
      throw new Error("arguments must have same shape!");
    this.saveForBackward([y.data, x.data]);
    return mathops.multiplyScale(x.data, y.data, 1);
  }

  backward(outputDerivative, argIndex) {
    var [ydata, xdata] = this.getSavedData();
    switch(argIndex) {
      case 0:
        return mathops.multiplyScale(outputDerivative, ydata, 1);
      case 1:
        return mathops.multiplyScale(outputDerivative, xdata, 1);
    }
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
    if(!mathops.sameShape(x.data, y.data))
      throw new Error("arguments must have same shape!");
    this.saveForBackward([y.data, x.data]);
    return mathops.divideScale(x.data, y.data, 1);
  }

  backward(outputDerivative, argIndex) {
    var [ydata, xdata] = this.getSavedData();
    switch(argIndex) {
      case 0:
        return mathops.divideScale(outputDerivative, ydata, 1);
      case 1:
        var answer_holder = mathops.multiplyScale(ydata, ydata, 1);
        var divided = mathops.divideScale(outputDerivative, answer_holder, -1, answer_holder);
        return mathops.multiplyScale(answer_holder, xdata, 1, answer_holder);
    }
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
    return mathops.scale(v.data, this.x);
  }

  backward(outputDerivative, argIndex) {
    return mathops.scale(outputDerivative, this.x);
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
    return mathops.addScale(v.data, this.x, 1);
  }

  backward(outputDerivative, argIndex) {
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
    return x.data.dot(y.data);
  }

  backward(outputDerivative, argIndex) {
    var [xdata , ydata] = this.getSavedData();
    switch(argIndex) {
      case 0:
        return mathops.multiplyScale(ydata, outputDerivative, 1);
      case 1:
        return mathops.multiplyScale(xdata, outputDerivative, 1);
    }
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
    return mathops.multiplyScale(x.data, x.data, 1);
  }

  backward(outputDerivative) {
    var inputData = this.getSavedData();
    return mathops.multiplyScale(outputDerivative, inputData, 2);
  }
}
exports.Square = Square;

function square(x) {
  return (new Square())(x);
}
exports.square = square;
exports.utilityFuncs.push(square);


class Exp extends autograd.Operation {

  forward(x) {
    var expX = mathops.exp(x.data);
    this.saveForBackward(expX);
    return expX;
  }

  backward(outputDerivative, argIndex) {
    var expX = this.getSavedData();
    var answer =  mathops.multiplyScale(outputDerivative, expX, 1);
    return answer;
  }
}
exports.Exp = Exp;

function exp(x) {
  return (new Exp())(x);
}
exports.exp = exp;
exports.utilityFuncs.push(exp);

class Sqrt extends autograd.Operation {

  forward(x) {
    var sqrtX = mathops.sqrt(x.data);
    this.saveForBackward(sqrtX);
    return sqrtX;
  }

  backward(outputDerivative, argIndex) {
    var sqrtX = this.getSavedData();
    return mathops.divideScale(outputDerivative, sqrtX, 0.5);
  }
}
exports.Sqrt = Sqrt;

function sqrt(x) {
  return (new Exp())(x);
}
exports.sqrt = sqrt;
exports.utilityFuncs.push(sqrt);

class Sin extends autograd.Operation {

  forward(x) {
    var X = x.data;
    this.saveForBackward(X);
    return mathops.sin(x.data);
  }

  backward(outputDerivative, argIndex) {
    var X = this.getSavedData();
    return mathops.multiplyScale(outputDerivative, mathops.cos(X), 1);
  }
}
exports.Sin = Sin;

function sin(x) {
  return (new Sin())(x);
}
exports.sin = sin;
exports.utilityFuncs.push(sin);


class Cos extends autograd.Operation {

  forward(x) {
    var X = x.data;
    this.saveForBackward(X);
    return mathops.cos(x.data);
  }

  backward(outputDerivative, argIndex) {
    var X = this.getSavedData();
    return mathops.multiplyScale(outputDerivative, mathops.sin(X), -1);
  }
}
exports.Cos = Cos;

function cos(x) {
  return (new Cos())(x);
}
exports.cos = cos;
exports.utilityFuncs.push(cos);

class Tan extends autograd.Operation {

  forward(x) {
    var tanX = mathops.tan(x.data);
    this.saveForBackward(tanX);
    return tanX;
  }

  backward(outputDerivative, argIndex) {
    var tanX = this.getSavedData();

    var tanXsquared = mathops.multiplyScale(tanX, tanX, 1);
    //re-use the storage of tanXsquared for the derivative.
    var secXsquared = mathops.addScale(tanXsquared, 1, 1, 1);
    var derivative = mathops.multiplyScale(outputDerivative,
                                       secXsquared,
                                       1,
                                       secXsquared);
    return derivative;
  }
}
exports.Tan = Tan;

function tan(x) {
  return (new Tan())(x);
}
exports.tan = tan;
exports.utilityFuncs.push(tan);

class Sum extends autograd.Operation {

  forward(x) {
    this.saveForBackward(x.data.shape);
    return x.data.sum();
  }

  backward(outputDerivative, argIndex) {
    var shape = this.getSavedData();
    return mathops.multiplyScale(outputDerivative,
                                tensor.onesLike(shape),
                                1);
  }
}
exports.Sum = Sum;

function sum(x) {
  return (new Sum())(x);
}
exports.sum = sum;
exports.utilityFuncs.push(sum);

class Log extends autograd.Operation {

  forward(x) {
    this.saveForBackward(x.data);
    return mathops.log(x.data);
  }

  backward(outputDerivative, argIndex) {
    var xdata = this.getSavedData();
    return mathops.divideScale(1, xdata, 1);
  }
}
exports.Log = Log;

function log(x) {
  return (new Log())(x);
}
exports.log = log;
exports.utilityFuncs.push(log);
