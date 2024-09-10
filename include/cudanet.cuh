#ifndef CUDANET_H
#define CUDANET_H

#ifdef USE_CUDA
#include "activation_functions.cuh"
#include "convolution.cuh"
#include "matmul.cuh"
#include "pooling.cuh"
#endif

// Layers
#include "activation.hpp"
#include "add.hpp"
#include "avg_pooling.hpp"
#include "batch_norm.cuh"
#include "concat.hpp"
#include "conv2d.cuh"
#include "dense.hpp"
#include "input.hpp"
#include "layer.hpp"
#include "max_pooling.hpp"
#include "output.hpp"

// Models
#include "model.hpp"
#include "module.hpp"

// Utils
#include "imagenet.hpp"
#ifdef USE_CUDA
#include "cuda_helper.cuh"
#include "vector.cuh"
#endif

#endif  // CUDANET_H