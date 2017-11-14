{
  "targets": [
    {
      "target_name": "tensorBinding",
      "sources": [
        "src/tensorBinding.cc",
        "src/tensor.cc",
        "src/mathops.cc"
        ],
      "cflags!": [
        "-fno-exceptions"
        ],
      "cflags_cc!": [
        "-fno-exceptions"
        ],
      "conditions": [
          ['OS=="mac"', {
            'xcode_settings': {
              'GCC_ENABLE_CPP_EXCEPTIONS': 'YES'
            }
          }]
        ],
      "cflags": [
        "-O3"
        ],
      "cflags_cc": [
        "-O3"
        ],
      "libraries": [
        "-lblas"
        ]
    }
  ]
}
