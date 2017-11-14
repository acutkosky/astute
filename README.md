# astute
a machine learning and tensor algebra package for node

This package requires BLAS to function. If you're on a mac it should be already present. Otherwise you'll need to find an installation.

### Tensor Algebra:

Declare a new tensor:
```
var astute = require('astute');
var Tensor = astute.tensor.Tensor;

var T = new Tensor({
  shape:[3,3],
  data: [1,2,3,
         4,5,6,
         7,8,9]
  });
```
If `data` is unspecified, the tensor will be filled with zeros.

You can declare multi-dimensional tensors:
```
var T = new Tensor({
  shape: [2, 2, 2],
  data: [ 1, 2, 3, 4   ,  5, 6, 7, 8,
          9 ,10,11,12  ,  13,14,15,16 ]
  });

//simpler declaration with nested arrays
var T = new Tensor([[1,2,3], 
                    [4,5,6]]);
```

Tensor contraction with `contract`:
```
var T1 = new Tensor({shape: [2,3,4]});
var T2 = new Tensor({shape:[4,3,5]});

var T3 = astute.tensor.contract(T1, T2, 2);
//T3 has shape [2, 5]

var T4 = T1.contract(T2, 1);
//T4 has shape [2, 3, 5]
```
Matrix multiplication with `matMul` (this is just a wrapper around `contract`):
```
var matrix1 = new Tensor({shape: [3,4]});
var matrix2 = new Tensor({shape: [4,5]});
var vector = new Tensor({shape: [5]});

var mm = astute.tensor.matMul(matrix1, matrix2);
var mv = matrix2.matMul(vector);
```
When possible, operations are performed using BLAS.

### Sparse Vectors
There is some support for sparse vectors (not sparse matrices or tensors though).
```
//create a 10000-dimensional sparse vector S with S[1] = 123  and S[34] = 23423 (S is 0-indexed).
var S = new astute.tensor.SparseVector([[1,123], [34, 23423]], 10000);

//do the same operation:
var m = new Map();
m.set(1, 123);
m.set(34, 23423);
var S = new astute.tensor.SparseVector(m, 10000);

//compute a dot-product:
var T = new astute.tensor.random.normalLike([1000], 0, 4);
var dotProduct = S.dot(T);
```
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

#### Optimizers:
Some handy optimizers are built in now. I'll probably add more later:
```
var astute = require('astute');

var {tensor, autograd, optim} = astute;

var eta = 1.0;
var x = new autograd.Variable([10, 8]);
var y = new autograd.Variable([3, -4], {requiresGrad: false});

var opt = new optim.AdaGrad(eta, [x]);

for(let t=1; t<100; t++) {
  //loss = ( <x,y> - 6 )^2
  loss = x.dot(y).sub(6).square();
  
  //computes gradients and takes a single optimization step:
  opt.step(loss);
}
```
You can create your own optimizers by subclassing `astute.optim.Optimizer`.


