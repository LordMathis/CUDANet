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
        const shape2d      inputSize,
        const int          inputChannels,
        const int          outputChannels,
        const shape2d      kernelSize,
        const shape2d      stride,
        const shape2d      padding,
        const std::string &prefix
    ) {
        // Create the convolution layer
        conv = new CUDANet::Layers::Conv2d(
            inputSize, inputChannels, kernelSize, stride, outputChannels,
            padding, CUDANet::Layers::ActivationType::NONE
        );

        shape2d batchNormSize = conv->getOutputDims();

        batchNorm = new CUDANet::Layers::BatchNorm2d(
            batchNormSize, outputChannels, 1e-3f,
            CUDANet::Layers::ActivationType::RELU
        );

        addLayer(prefix + ".conv", conv);
        addLayer(prefix + ".bn", batchNorm);
    }

    ~BasicConv2d() {
        delete conv;
        delete batchNorm;
    }

    float *forward(const float *d_input) {
        float *d_output = conv->forward(d_input);
        return batchNorm->forward(d_output);
    }

    shape2d getOutputDims() {
        return batchNorm->getOutputDims();
    }

  private:
    CUDANet::Layers::Conv2d      *conv;
    CUDANet::Layers::BatchNorm2d *batchNorm;
};

class InceptionA : public CUDANet::Module {
  public:
    InceptionA(
        const shape2d      inputSize,
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
            inputSize, inputChannels, {3, 3}, {1, 1}, {1, 1},
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

    ~InceptionA() {
        delete branch1x1;
        delete branch5x5_1;
        delete branch5x5_2;
        delete branch3x3dbl_1;
        delete branch3x3dbl_2;
        delete branch3x3dbl_3;
        delete branchPool_1;
        delete branchPool_2;
        delete concat_1;
        delete concat_2;
        delete concat_3;
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
    shape2d inputSize;
    int     inputChannels;
    int     poolFeatures;

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
        const shape2d      inputSize,
        const int          inputChannels,
        const std::string &prefix
    )
        : inputSize(inputSize), inputChannels(inputChannels) {
        // Branch 3x3
        branch3x3 = new BasicConv2d(
            inputSize, inputChannels, 384, {3, 3}, {2, 2}, {0, 0},
            prefix + ".branch1x1"
        );
        addLayer("", branch3x3);

        // Branch 3x3dbl
        branch3x3dbl_1 = new BasicConv2d(
            inputSize, inputChannels, 64, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch3x3dbl_1"
        );
        addLayer("", branch3x3dbl_1);
        branch3x3dbl_2 = new BasicConv2d(
            branch3x3dbl_1->getOutputDims(), 96, 96, {3, 3}, {1, 1}, {1, 1},
            prefix + ".branch3x3dbl_2"
        );
        addLayer("", branch3x3dbl_2);
        branch3x3dbl_3 = new BasicConv2d(
            branch3x3dbl_2->getOutputDims(), 96, 96, {3, 3}, {2, 2}, {1, 1},
            prefix + ".branch3x3dbl_3"
        );
        addLayer("", branch3x3dbl_3);

        branchPool = new CUDANet::Layers::MaxPooling2d(
            inputSize, inputChannels, {3, 3}, {2, 2}, {0, 0},
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

    ~InceptionB() {
        delete branch3x3;
        delete branch3x3dbl_1;
        delete branch3x3dbl_2;
        delete branch3x3dbl_3;
        delete branchPool;
        delete concat_1;
        delete concat_2;
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
    shape2d inputSize;
    int     inputChannels;

    BasicConv2d *branch3x3;

    BasicConv2d *branch3x3dbl_1;
    BasicConv2d *branch3x3dbl_2;
    BasicConv2d *branch3x3dbl_3;

    CUDANet::Layers::MaxPooling2d *branchPool;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
};

class InceptionC : public CUDANet::Module {
  public:
    InceptionC(
        const shape2d      inputSize,
        const int          inputChannels,
        const int          nChannels_7x7,
        const std::string &prefix
    )
        : inputSize(inputSize), inputChannels(inputChannels) {
        // Branch 1x1
        branch1x1 = new BasicConv2d(
            inputSize, inputChannels, 192, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch1x1"
        );
        addLayer("", branch1x1);

        // Branch 7x7
        branch7x7_1 = new BasicConv2d(
            inputSize, inputChannels, nChannels_7x7, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch7x7_1"
        );
        addLayer("", branch7x7_1);
        branch7x7_2 = new BasicConv2d(
            branch7x7_1->getOutputDims(), nChannels_7x7, nChannels_7x7, {1, 7},
            {1, 1}, {0, 3}, prefix + ".branch7x7_2"
        );
        addLayer("", branch7x7_2);
        branch7x7_3 = new BasicConv2d(
            branch7x7_2->getOutputDims(), nChannels_7x7, 192, {7, 1}, {1, 1},
            {3, 0}, prefix + ".branch7x7_3"
        );
        addLayer("", branch7x7_3);

        // Branch 7x7dbl
        branch7x7dbl_1 = new BasicConv2d(
            inputSize, inputChannels, nChannels_7x7, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch7x7dbl_1"
        );
        addLayer("", branch7x7dbl_1);
        branch7x7dbl_2 = new BasicConv2d(
            branch7x7dbl_1->getOutputDims(), nChannels_7x7, nChannels_7x7,
            {7, 1}, {1, 1}, {3, 0}, prefix + ".branch7x7dbl_2"
        );
        addLayer("", branch7x7dbl_2);
        branch7x7dbl_3 = new BasicConv2d(
            branch7x7dbl_2->getOutputDims(), nChannels_7x7, nChannels_7x7,
            {1, 7}, {1, 1}, {0, 3}, prefix + ".branch7x7dbl_3"
        );
        addLayer("", branch7x7dbl_3);
        branch7x7dbl_4 = new BasicConv2d(
            branch7x7dbl_3->getOutputDims(), nChannels_7x7, nChannels_7x7,
            {7, 1}, {1, 1}, {3, 0}, prefix + ".branch7x7dbl_4"
        );
        addLayer("", branch7x7dbl_4);
        branch7x7dbl_5 = new BasicConv2d(
            branch7x7dbl_4->getOutputDims(), nChannels_7x7, 192, {1, 7}, {1, 1},
            {0, 3}, prefix + ".branch7x7dbl_5"
        );
        addLayer("", branch7x7dbl_5);

        // Branch Pool
        branchPool_1 = new CUDANet::Layers::AvgPooling2d(
            inputSize, inputChannels, {3, 3}, {1, 1}, {1, 1},
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("", branchPool_1);
        branchPool_2 = new BasicConv2d(
            branchPool_1->getOutputDims(), inputChannels, 192, {1, 1}, {1, 1},
            {0, 0}, prefix + ".branchPool_2"
        );
        addLayer("", branchPool_2);

        // Concat
        concat_1 = new CUDANet::Layers::Concat(
            branch1x1->getOutputSize(), branch7x7_3->getOutputSize()
        );
        concat_2 = new CUDANet::Layers::Concat(
            concat_1->getOutputSize(), branch7x7dbl_5->getOutputSize()
        );
        concat_3 = new CUDANet::Layers::Concat(
            concat_2->getOutputSize(), branchPool_2->getOutputSize()
        );
    }

    ~InceptionC() {
        delete branch1x1;
        delete branch7x7_1;
        delete branch7x7_2;
        delete branch7x7_3;
        delete branch7x7dbl_1;
        delete branch7x7dbl_2;
        delete branch7x7dbl_3;
        delete branch7x7dbl_4;
        delete branch7x7dbl_5;
        delete branchPool_1;
        delete branchPool_2;
        delete concat_1;
        delete concat_2;
        delete concat_3;
    }

    float *forward(const float *d_input) {
        float *branch1x1_output = branch1x1->forward(d_input);

        float *branch7x7_output = branch7x7_1->forward(d_input);
        branch7x7_output        = branch7x7_2->forward(branch7x7_output);
        branch7x7_output        = branch7x7_3->forward(branch7x7_output);

        float *branch7x7dbl_output = branch7x7dbl_1->forward(d_input);
        branch7x7dbl_output = branch7x7dbl_2->forward(branch7x7dbl_output);
        branch7x7dbl_output = branch7x7dbl_3->forward(branch7x7dbl_output);
        branch7x7dbl_output = branch7x7dbl_4->forward(branch7x7dbl_output);
        branch7x7dbl_output = branch7x7dbl_5->forward(branch7x7dbl_output);

        float *branchPool_output = branchPool_1->forward(d_input);
        branchPool_output        = branchPool_2->forward(branchPool_output);

        float *d_output = concat_1->forward(branch1x1_output, branch7x7_output);
        d_output        = concat_2->forward(d_output, branch7x7dbl_output);
        d_output        = concat_3->forward(d_output, branchPool_output);

        return d_output;
    }

  private:
    shape2d inputSize;
    int     inputChannels;

    BasicConv2d *branch1x1;

    BasicConv2d *branch7x7_1;
    BasicConv2d *branch7x7_2;
    BasicConv2d *branch7x7_3;

    BasicConv2d *branch7x7dbl_1;
    BasicConv2d *branch7x7dbl_2;
    BasicConv2d *branch7x7dbl_3;
    BasicConv2d *branch7x7dbl_4;
    BasicConv2d *branch7x7dbl_5;

    CUDANet::Layers::AvgPooling2d *branchPool_1;
    BasicConv2d                   *branchPool_2;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
    CUDANet::Layers::Concat *concat_3;
};

class InceptionD : public CUDANet::Module {
  public:
    InceptionD(
        const shape2d      inputSize,
        const int          inputChannels,
        const std::string &prefix
    )
        : inputSize(inputSize), inputChannels(inputChannels) {
        // Branch 3x3
        branch3x3_1 = new BasicConv2d(
            inputSize, inputChannels, 192, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch3x3"
        );
        addLayer("", branch3x3_1);
        branch3x3_2 = new BasicConv2d(
            inputSize, 192, 320, {3, 3}, {2, 2}, {0, 0}, prefix + ".branch3x3_2"
        );
        addLayer("", branch3x3_2);

        // Branch 7x7x3
        branch7x7x3_1 = new BasicConv2d(
            inputSize, inputChannels, 192, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch7x7x3_1"
        );
        addLayer("", branch7x7x3_1);
        branch7x7x3_2 = new BasicConv2d(
            inputSize, 192, 192, {1, 7}, {1, 1}, {0, 3},
            prefix + ".branch7x7x3_2"
        );
        addLayer("", branch7x7x3_2);
        branch7x7x3_3 = new BasicConv2d(
            inputSize, 192, 192, {7, 1}, {1, 1}, {3, 0},
            prefix + ".branch7x7x3_3"
        );
        addLayer("", branch7x7x3_3);
        branch7x7x3_4 = new BasicConv2d(
            inputSize, 192, 192, {3, 3}, {2, 2}, {0, 0},
            prefix + ".branch7x7x3_4"
        );
        addLayer("", branch7x7x3_4);

        // Branch Pool
        branchPool = new CUDANet::Layers::MaxPooling2d(
            inputSize, 192, {3, 3}, {2, 2}, {0, 0},
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("", branchPool);

        // Concat
        concat_1 = new CUDANet::Layers::Concat(
            branch3x3_2->getOutputSize(), branch7x7x3_4->getOutputSize()
        );
        concat_2 = new CUDANet::Layers::Concat(
            concat_1->getOutputSize(), branchPool->getOutputSize()
        );
    }

    ~InceptionD() {
        delete branch3x3_1;
        delete branch3x3_2;
        delete branch7x7x3_1;
        delete branch7x7x3_2;
        delete branch7x7x3_3;
        delete branch7x7x3_4;
        delete branchPool;
        delete concat_1;
        delete concat_2;
    }

    float *forward(float *d_input) {
        float *branch1x1_output = branch3x3_1->forward(d_input);
        branch1x1_output        = branch3x3_2->forward(branch1x1_output);

        float *branch7x7_output = branch7x7x3_1->forward(d_input);
        branch7x7_output        = branch7x7x3_2->forward(branch7x7_output);
        branch7x7_output        = branch7x7x3_3->forward(branch7x7_output);
        branch7x7_output        = branch7x7x3_4->forward(branch7x7_output);

        float *branchPool_output = branchPool->forward(d_input);

        float *d_output = concat_1->forward(branch1x1_output, branch7x7_output);
        d_output        = concat_2->forward(d_output, branchPool_output);

        return d_output;
    }

  private:
    shape2d inputSize;
    int     inputChannels;

    BasicConv2d *branch3x3_1;
    BasicConv2d *branch3x3_2;

    BasicConv2d *branch7x7x3_1;
    BasicConv2d *branch7x7x3_2;
    BasicConv2d *branch7x7x3_3;
    BasicConv2d *branch7x7x3_4;

    CUDANet::Layers::MaxPooling2d *branchPool;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
};

class InceptionE : public CUDANet::Module {
  public:
    InceptionE(
        const shape2d      inputSize,
        const int          inputChannels,
        const std::string &prefix
    )
        : inputSize(inputSize), inputChannels(inputChannels) {
        // Branch 1x1
        branch1x1 = new BasicConv2d(
            inputSize, inputChannels, 320, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch1x1"
        );
        addLayer("", branch1x1);

        // Branch 3x3
        branch3x3_1 = new BasicConv2d(
            inputSize, inputChannels, 384, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch3x3_1"
        );
        addLayer("", branch3x3_1);
        branch3x3_2a = new BasicConv2d(
            inputSize, 384, 384, {1, 3}, {1, 1}, {0, 1},
            prefix + ".branch3x3_2a"
        );
        addLayer("", branch3x3_2a);
        branch3x3_2b = new BasicConv2d(
            inputSize, 384, 384, {3, 1}, {1, 1}, {1, 0},
            prefix + ".branch3x3_2b"
        );
        addLayer("", branch3x3_2b);
        branch_3x3_2_concat = new CUDANet::Layers::Concat(
            branch3x3_2a->getOutputSize(), branch3x3_2b->getOutputSize()
        );

        // Branch 3x3dbl
        branch3x3dbl_1 = new BasicConv2d(
            inputSize, inputChannels, 448, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch3x3dbl_1"
        );
        addLayer("", branch3x3dbl_1);
        branch3x3dbl_2 = new BasicConv2d(
            inputSize, 448, 384, {3, 3}, {1, 1}, {1, 1},
            prefix + ".branch3x3dbl_2"
        );
        addLayer("", branch3x3dbl_2);
        branch3x3dbl_3a = new BasicConv2d(
            inputSize, 384, 384, {1, 3}, {1, 1}, {0, 1},
            prefix + ".branch3x3dbl_3a"
        );
        addLayer("", branch3x3dbl_3a);
        branch3x3dbl_3b = new BasicConv2d(
            inputSize, 384, 384, {3, 1}, {1, 1}, {1, 0},
            prefix + ".branch3x3dbl_3b"
        );
        addLayer("", branch3x3dbl_3b);
        branch_3x3dbl_3_concat = new CUDANet::Layers::Concat(
            branch3x3dbl_3a->getOutputSize(), branch3x3dbl_3b->getOutputSize()
        );

        // Branch Pool
        branchPool_1 = new CUDANet::Layers::AvgPooling2d(
            inputSize, inputChannels, {3, 3}, {1, 1}, {1, 1},
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("", branchPool_1);
        branchPool_2 = new BasicConv2d(
            inputSize, inputChannels, 192, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branchPool_2"
        );
        addLayer("", branchPool_2);

        // Concat
        concat_1 = new CUDANet::Layers::Concat(
            branch1x1->getOutputSize(), branch_3x3_2_concat->getOutputSize()
        );
        concat_2 = new CUDANet::Layers::Concat(
            concat_1->getOutputSize(), branch_3x3dbl_3_concat->getOutputSize()
        );
        concat_3 = new CUDANet::Layers::Concat(
            concat_2->getOutputSize(), branchPool_2->getOutputSize()
        );
    }

    ~InceptionE() {
        delete branch1x1;
        delete branch3x3_1;
        delete branch3x3_2a;
        delete branch3x3_2b;
        delete branch_3x3_2_concat;
        delete branch3x3dbl_1;
        delete branch3x3dbl_2;
        delete branch3x3dbl_3a;
        delete branch3x3dbl_3b;
        delete branch_3x3dbl_3_concat;
        delete branchPool_1;
        delete branchPool_2;
        delete concat_1;
        delete concat_2;
        delete concat_3;
    }

    float *forward(const float *d_input) {
        float *branch1x1_output = branch1x1->forward(d_input);

        float *branch3x3_output    = branch3x3_1->forward(d_input);
        float *branch3x3_2a_output = branch3x3_2a->forward(branch3x3_output);
        float *branch3x3_2b_output = branch3x3_2b->forward(branch3x3_output);
        branch3x3_output           = branch_3x3_2_concat->forward(
            branch3x3_2a_output, branch3x3_2b_output
        );

        float *branch3x3dbl_output = branch3x3dbl_1->forward(d_input);
        branch3x3dbl_output = branch3x3dbl_2->forward(branch3x3dbl_output);
        float *branch3x3dbl_3a_output =
            branch3x3dbl_3a->forward(branch3x3dbl_output);
        float *branch3x3dbl_3b_output =
            branch3x3dbl_3b->forward(branch3x3dbl_output);
        branch3x3dbl_output = branch_3x3dbl_3_concat->forward(
            branch3x3dbl_3a_output, branch3x3dbl_3b_output
        );

        float *branchPool_output = branchPool_1->forward(d_input);
        branchPool_output        = branchPool_2->forward(branchPool_output);

        float *d_output = concat_1->forward(branch1x1_output, branch3x3_output);
        d_output        = concat_2->forward(d_output, branch3x3dbl_output);
        d_output        = concat_3->forward(d_output, branchPool_output);

        return d_output;
    }

  private:
    shape2d inputSize;
    int     inputChannels;

    BasicConv2d *branch1x1;

    BasicConv2d             *branch3x3_1;
    BasicConv2d             *branch3x3_2a;
    BasicConv2d             *branch3x3_2b;
    CUDANet::Layers::Concat *branch_3x3_2_concat;

    BasicConv2d             *branch3x3dbl_1;
    BasicConv2d             *branch3x3dbl_2;
    BasicConv2d             *branch3x3dbl_3a;
    BasicConv2d             *branch3x3dbl_3b;
    CUDANet::Layers::Concat *branch_3x3dbl_3_concat;

    CUDANet::Layers::AvgPooling2d *branchPool_1;
    BasicConv2d                   *branchPool_2;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
    CUDANet::Layers::Concat *concat_3;
};