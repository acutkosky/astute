# astute
a machine learning and tensor algebra package for node

This package requires BLAS to function. If you're on a mac it should be already present. Otherwise you'll need to find an installation.

### Tensor Algebra:

Declare a new tensor:
```
var astute = require('astute');

var T = new astute.Tensor({
  shape:[3,3],
  data: [1,2,3,
         4,5,6,
         7,8,9]
  });
```
If `data` is unspecified, the tensor will be filled with zeros.

You can declare multi-dimensional tensors:
```
var T = new astute.Tensor({
  shape: [2, 2, 2],
  data: [ 1, 2, 3, 4   ,  5, 6, 7, 8,
          9 ,10,11,12  ,  13,14,15,16 ]
  });
```

Tensor contraction with `contract`:
```
var T1 = new astute.Tensor({shape: [2,3,4]});
var T2 = new astute.Tensor({shape:[4,3,5]});

var T3 = astute.contract(T1, T2, 2);
//T3 has shape [2, 5]

var T4 = T1.contract(T2, 1);
//T4 has shape [2, 3, 5]
```
Matrix multiplication with `matMul` (this is just a wrapper around `contract`):
```
var matrix1 = new astute.Tensor({shape: [3,4]});
var matrix2 = new astute.Tensor({shape: [4,5]});
var vector = new astute.Tensor({shape: [5]});

var mm = astute.matMul(matrix1, matrix2);
var mv = matrix2.matMul(vector);
```
When possible, operations are performed using BLAS.

### Automatic Differentation:
`astute.autograd` contains procedures for automatic differentation. It's modeled after Pytorch's module of the same name. Here's a bare-bones example:
```
var astute = require('astute');

var {tensor, autograd} = astute;

var eta = 0.1;
var x = new autograd.Variable(10);
var y = new autograd.Variable(3);

for(let t=1; t<100; t++) {
  //loss = ( <x,y> - 6 )^2
  loss = x.dot(y).sub(6).square();

  //compute gradients
  loss.zeroGrad();
  loss.backward(1.0);
  //x.data is a Tensor object containing the current value of x.
  //x.grad is a Tensor object that now contains the value of dloss/dx.
  
  //manual implementation of a gradient descent update:
  //x.data = 1.0*x.data + (-eta/sqrt(t))*x.grad;
  x.data = tensor.addScale(x.data, x.grad, 1.0, -eta/Math.sqrt(t));
}
```

