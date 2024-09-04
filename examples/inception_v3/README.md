# Inception v3

Inception v3 Inference on CUDANet

## Usage

1. Export pytorch Inception v3 weights pretrained on ImageNet (requires pytorch and torchvision):

```sh
python inception_v3.py
```

2. Follow the instructions from the repository root to build the CUDANet library.

3. Build Inception v3 (requires [OpenCV](https://opencv.org/) for image loading and preprocessing):

```sh
mkdir build
cd build
cmake -S ..
make
```

4. (Optional) Run tests

Generate test input/output and resources by running `inception_blocks.py` in the `test` folder

Build and run tests (requires [Google Test](https://github.com/google/googletest))

```sh
cd build
make test_inception_v3
./tests/test_inception_v3
```

5. Run Inception v3 inference:
```sh
inception_v3 ../inception_v3_weights.bin ../image.jpg
```

## Note on Preprocessing

The image preprocessing in this implementation uses OpenCV, which may produce slightly different results compared to PyTorch's Pillow-based preprocessing due to differences in interpolation methods during resizing.
