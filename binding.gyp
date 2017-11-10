{
  "targets": [
    {
      "target_name": "tensorBinding",
      "sources": [
        "src/tensorBinding.cc",
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
