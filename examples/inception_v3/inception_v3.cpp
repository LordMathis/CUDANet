#include <cudanet.cuh>
#include <iostream>

int main(int argc, const char *const argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << "<model_weights_path> <image_path>"
                  << std::endl;
        return 1;  // Return error code indicating incorrect usage
    }

    std::cout << "Loading model..." << std::endl;
}

class BasicConv2d : public CUDANet::Module {
  public:
    BasicConv2d(
        const dim2d        inputSize,
        const int          inputChannels,
        const int          outputChannels,
        const dim2d        kernelSize,
        const dim2d        stride,
        const dim2d        padding,
        const std::string &prefix
    ) {
        // Create the convolution layer
        conv = new CUDANet::Layers::Conv2d(
            inputSize, inputChannels, kernelSize, stride, outputChannels,
            padding, CUDANet::Layers::ActivationType::NONE
        );

        dim2d batchNormSize = conv->getOutputDims();

        batchNorm = new CUDANet::Layers::BatchNorm2d(
            batchNormSize, outputChannels, 1e-3f,
            CUDANet::Layers::ActivationType::RELU
        );

        addLayer(prefix + ".conv", conv);
        addLayer(prefix + ".bn", batchNorm);
    }

    float *forward(const float *d_input) {
        float *d_output = conv->forward(d_input);
        return batchNorm->forward(d_output);
    }

    dim2d getOutputDims() {
        return batchNorm->getOutputDims();
    }

  private:
    CUDANet::Layers::Conv2d      *conv;
    CUDANet::Layers::BatchNorm2d *batchNorm;
};

class InceptionA : public CUDANet::Module {
  public:
    InceptionA(
        const dim2d        inputSize,
        const int          inputChannels,
        const int          poolFeatures,
        const std::string &prefix
    )
        : inputSize(inputSize),
          inputChannels(inputChannels),
          poolFeatures(poolFeatures) {
        // Branch 1x1
        branch1x1 = new BasicConv2d(
            inputSize, inputChannels, 64, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch1x1"
        );
        addLayer("", branch1x1);

        // Branch 5x5
        branch5x5_1 = new BasicConv2d(
            inputSize, inputChannels, 48, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch5x5_1"
        );
        addLayer("", branch5x5_1);
        branch5x5_2 = new BasicConv2d(
            inputSize, 48, 64, {5, 5}, {1, 1}, {2, 2}, prefix + ".branch5x5_2"
        );
        addLayer("", branch5x5_2);

        // Branch 3x3
        branch3x3dbl_1 = new BasicConv2d(
            inputSize, inputChannels, 64, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch3x3dbl_1"
        );
        addLayer("", branch3x3dbl_1);
        branch3x3dbl_2 = new BasicConv2d(
            inputSize, 64, 96, {3, 3}, {1, 1}, {1, 1},
            prefix + ".branch3x3dbl_2"
        );
        addLayer("", branch3x3dbl_2);
        branch3x3dbl_3 = new BasicConv2d(
            inputSize, 96, 96, {3, 3}, {1, 1}, {1, 1},
            prefix + ".branch3x3dbl_3"
        );
        addLayer("", branch3x3dbl_3);

        // Branch Pool
        branchPool_1 = new CUDANet::Layers::AvgPooling2d(
            inputSize, inputChannels, {3, 3}, {1, 1},
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("", branchPool_1);
        branchPool_2 = new BasicConv2d(
            branchPool_1->getOutputDims(), inputChannels, poolFeatures, {1, 1},
            {1, 1}, {0, 0}, prefix + ".branchPool"
        );
        addLayer("", branchPool_2);

        // Concat
        concat_1 = new CUDANet::Layers::Concat(
            branch1x1->getOutputSize(), branch5x5_2->getOutputSize()
        );
        concat_2 = new CUDANet::Layers::Concat(
            concat_1->getOutputSize(), branch3x3dbl_3->getOutputSize()
        );
        concat_3 = new CUDANet::Layers::Concat(
            concat_2->getOutputSize(), branchPool_2->getOutputSize()
        );
    }

    float *forward(const float *d_input) {
        float *d_branch1x1_out = branch1x1->forward(d_input);

        float *d_branch5x5_out = branch5x5_1->forward(d_input);
        d_branch5x5_out        = branch5x5_2->forward(d_branch5x5_out);

        float *d_branch3x3_out = branch3x3dbl_1->forward(d_input);
        d_branch3x3_out        = branch3x3dbl_2->forward(d_branch3x3_out);
        d_branch3x3_out        = branch3x3dbl_3->forward(d_branch3x3_out);

        float *d_branchPool_out = branchPool_1->forward(d_input);
        d_branchPool_out        = branchPool_2->forward(d_branchPool_out);

        float *d_output = concat_1->forward(d_branch1x1_out, d_branch5x5_out);
        d_output        = concat_2->forward(d_output, d_branch3x3_out);
        d_output        = concat_3->forward(d_output, d_branchPool_out);

        return d_output;
    }

  private:
    dim2d inputSize;
    int   inputChannels;
    int   poolFeatures;

    BasicConv2d *branch1x1;

    BasicConv2d *branch5x5_1;
    BasicConv2d *branch5x5_2;

    BasicConv2d *branch3x3dbl_1;
    BasicConv2d *branch3x3dbl_2;
    BasicConv2d *branch3x3dbl_3;

    CUDANet::Layers::AvgPooling2d *branchPool_1;
    BasicConv2d                   *branchPool_2;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
    CUDANet::Layers::Concat *concat_3;
};

class InceptionB : public CUDANet::Module {
  public:
    InceptionB(
        const dim2d        inputSize,
        const int          inputChannels,
        const std::string &prefix
    )
        : inputSize(inputSize), inputChannels(inputChannels) {
        // Branch 3x3
        branch3x3 = new BasicConv2d(
            inputSize, inputChannels, 384, {3, 3}, {2, 2}, {0, 0}, "branch1x1"
        );
        addLayer("", branch3x3);

        // Branch 3x3dbl
        branch3x3dbl_1 = new BasicConv2d(
            inputSize, inputChannels, 64, {1, 1}, {1, 1}, {0, 0},
            "branch3x3dbl_1"
        );
        addLayer("", branch3x3dbl_1);
        branch3x3dbl_2 = new BasicConv2d(
            branch3x3dbl_1->getOutputDims(), 96, 96, {3, 3}, {1, 1}, {1, 1},
            "branch3x3dbl_2"
        );
        addLayer("", branch3x3dbl_2);
        branch3x3dbl_3 = new BasicConv2d(
            branch3x3dbl_2->getOutputDims(), 96, 96, {3, 3}, {2, 2}, {1, 1},
            "branch3x3dbl_3"
        );
        addLayer("", branch3x3dbl_3);

        branchPool = new CUDANet::Layers::MaxPooling2d(
            inputSize, inputChannels, {3, 3}, {2, 2},
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer(prefix + ".branchPool", branchPool);

        concat_1 = new CUDANet::Layers::Concat(
            branch3x3->getOutputSize(), branch3x3dbl_3->getOutputSize()
        );
        concat_2 = new CUDANet::Layers::Concat(
            concat_1->getOutputSize(), branchPool->getOutputSize()
        );
    }

    float *forward(const float *d_input) {
        float *d_branch3x3_out = branch3x3->forward(d_input);

        float *d_branch3x3dbl_out = branch3x3dbl_1->forward(d_input);
        d_branch3x3dbl_out        = branch3x3dbl_2->forward(d_branch3x3dbl_out);
        d_branch3x3dbl_out        = branch3x3dbl_3->forward(d_branch3x3dbl_out);

        float *d_branchPool_out = branchPool->forward(d_input);

        float *d_output =
            concat_1->forward(d_branch3x3_out, d_branch3x3dbl_out);
        d_output = concat_2->forward(d_output, d_branchPool_out);

        return d_output;
    }

  private:
    dim2d inputSize;
    int   inputChannels;

    BasicConv2d *branch3x3;

    BasicConv2d *branch3x3dbl_1;
    BasicConv2d *branch3x3dbl_2;
    BasicConv2d *branch3x3dbl_3;

    CUDANet::Layers::MaxPooling2d *branchPool;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
};