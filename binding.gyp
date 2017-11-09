{
  "targets": [
    {
      "target_name": "tensor",
      "sources": [ 
        "src/node_tensor.cc",
        "src/tensor.cc",
        "src/mathops.cc"
        ],
      "cflags": [
        "-O3"
        ],
      "libraries": [
        "-lblas"
        ]
    }
  ]
}
