# CUDANet

:warning: Work in progress

Convolutional Neural Network inference library running on CUDA.

## Features

- [x] Input layer
- [x] Dense (fully-connected) layer
- [x] Conv2d layer
- [ ] Max pooling
- [ ] Average pooling
- [ ] Concat layer
- [x] Sigmoid activation
- [x] ReLU activation
- [x] Softmax activation
- [ ] Load weights from file 

## Usage

**requirements**
- [cmake](https://cmake.org/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Google Test](https://github.com/google/googletest) (for testing only)

**build**

```sh
mkdir build
cd build
cmake -S ..
make
```

**build and run tests**

```sh
make test_main
./test/test_main
```