# astute
a machine learning and tensor algebra package for node

This package requires BLAS to function. If you're on a mac it should be already present.

Tensor Algebra:

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

