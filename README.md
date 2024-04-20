# CUDANet

:warning: Work in progress

Convolutional Neural Network inference library running on CUDA.

## Features

- [x] Input layer
- [x] Dense (fully-connected) layer
- [x] Conv2d layer
- [x] Max pooling
- [x] Average pooling
- [x] Concat layer
- [x] Sigmoid activation
- [x] ReLU activation
- [x] Softmax activation
- [x] Load weights from file 

## Usage

**requirements**
- [cmake](https://cmake.org/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Google Test](https://github.com/google/googletest) (for testing only)

**build**

```sh
mkdir build
cd build
cmake -S .. -DCMAKE_CUDA_ARCHITECTURES=75  # Replace with you cuda architecture
make
```

**build and run tests**

```sh
make test_main
./test/test_main
```

### Create Layers and Model

```cpp
CUDANet::Model *model =
    new CUDANet::Model(inputSize, inputChannels, outputSize);

// Conv2d
CUDANet::Layers::Conv2d *conv2d = new CUDANet::Layers::Conv2d(
    inputSize, inputChannels, kernelSize, stride, numFilters,
    CUDANet::Layers::Padding::VALID,
    CUDANet::Layers::ActivationType::NONE
);

if (setWeights) {
    conv2d->setWeights(getConv1Weights().data());
}
model->addLayer("conv1", conv2d);
```

### Sequential and Functional API

Run prediction by passing the input through the layers in the order they have been added.

```cpp
std::vector<float> input = {...};
model->predict(input.data());
```

If you want to use more complex forward pass, using `Concat` or `Add` layers, you can subclass the model class and override the default `predict` function

```cpp
class MyModel : public CUDANet::Model {
    ...
}

...

float* MyModel::predict(const float* input) {
    float* d_input = inputLayer->forward(input);

    d_conv1 = getLayer("conv1")->forward(d_input);
    d_conv2 = getLayer("conv2")->forward(d_input);

    d_output = concatLayer->forward(d_conv1, d_conv2);

    return outputLayer->forward(d_input);
}
```

### Load Pre-trained Weights

CUDANet uses format similar to safetensors to load weights and biases.

```
[int64 header size, header, tensor values]
```

where `header` is a csv format

```
<tensor_name>,<tensor_size>,<tensor_offset>
```

To load weights call `load_weights` function on Model object.

To export weights from pytorch you can use the `export_model_weights` function from `tools/utils.py`  script