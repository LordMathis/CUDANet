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

    int getOutputChannels() {
        return outputChannels;
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
            branch5x5_1->getOutputDims(), 48, 64, {5, 5}, {1, 1}, {2, 2},
            prefix + ".branch5x5_2"
        );
        addLayer("", branch5x5_2);

        // Branch 3x3
        branch3x3dbl_1 = new BasicConv2d(
            inputSize, inputChannels, 64, {1, 1}, {1, 1}, {0, 0},
            prefix + ".branch3x3dbl_1"
        );
        addLayer("", branch3x3dbl_1);
        branch3x3dbl_2 = new BasicConv2d(
            branch3x3dbl_1->getOutputDims(), 64, 96, {3, 3}, {1, 1}, {1, 1},
            prefix + ".branch3x3dbl_2"
        );
        addLayer("", branch3x3dbl_2);
        branch3x3dbl_3 = new BasicConv2d(
            branch3x3dbl_2->getOutputDims(), 96, 96, {3, 3}, {1, 1}, {1, 1},
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

    shape2d getOutputDims() {
        return branch1x1->getOutputDims();
    }

    int getOutputChannels() {
        return branch1x1->getOutputChannels() +
               branch5x5_2->getOutputChannels() +
               branch3x3dbl_3->getOutputChannels() +
               branchPool_2->getOutputChannels();
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

    shape2d getOutputDims() {
        return branch3x3->getOutputDims();
    }

    int getOutputChannels() {
        return branch3x3->getOutputChannels() +
               branch3x3dbl_3->getOutputChannels() + inputChannels;
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

    shape2d getOutputDims() {
        return branch1x1->getOutputDims();
    }

    int getOutputChannels() {
        return branch1x1->getOutputChannels() +
               branch7x7_3->getOutputChannels() +
               branch7x7dbl_5->getOutputChannels() +
               branchPool_2->getOutputChannels();
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

    float *forward(const float *d_input) {
        float *branch3x3_output = branch3x3_1->forward(d_input);
        branch3x3_output        = branch3x3_2->forward(branch3x3_output);

        float *branch7x7_output = branch7x7x3_1->forward(d_input);
        branch7x7_output        = branch7x7x3_2->forward(branch7x7_output);
        branch7x7_output        = branch7x7x3_3->forward(branch7x7_output);
        branch7x7_output        = branch7x7x3_4->forward(branch7x7_output);

        float *branchPool_output = branchPool->forward(d_input);

        float *d_output = concat_1->forward(branch3x3_output, branch7x7_output);
        d_output        = concat_2->forward(d_output, branchPool_output);

        return d_output;
    }

    shape2d getOutputDims() {
        return branch3x3_2->getOutputDims();
    }

    int getOutputChannels() {
        return branch3x3_2->getOutputChannels() +
               branch7x7x3_4->getOutputChannels() + inputChannels;
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

    shape2d getOutputDims() {
        branch3x3_2a->getOutputDims();
    }

    int getOutputChannels() {
        return branch3x3_2a->getOutputChannels() +
               branch3x3_2b->getOutputChannels() +
               branch3x3dbl_3a->getOutputChannels() +
               branch3x3dbl_3b->getOutputChannels() +
               branchPool_2->getOutputChannels();
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

class InceptionV3 : public CUDANet::Model {
  public:
    InceptionV3(
        const shape2d inputSize,
        const int     inputChannels,
        const int     outputSize
    )
        : CUDANet::Model(inputSize, inputChannels, outputSize) {
        conv2d_1a_3x3 = new BasicConv2d(
            inputSize, inputChannels, 32, {3, 3}, {2, 2}, {0, 0},
            "conv2d_1a_3x3"
        );
        addLayer("", conv2d_1a_3x3);
        conv2d_2a_3x3 = new BasicConv2d(
            conv2d_1a_3x3->getOutputDims(), 32, 32, {3, 3}, {1, 1}, {0, 0},
            "conv2d_2a_3x3"
        );
        addLayer("", conv2d_2a_3x3);
        conv2d_2b_3x3 = new BasicConv2d(
            conv2d_2a_3x3->getOutputDims(), 32, 64, {3, 3}, {1, 1}, {1, 1},
            "conv2d_2b_3x3"
        );
        addLayer("", conv2d_2b_3x3);

        maxpool1 = new CUDANet::Layers::MaxPooling2d(
            conv2d_2b_3x3->getOutputDims(), 64, {3, 3}, {2, 2}, {0, 0},
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("maxpool1", maxpool1);

        conv2d_3b_1x1 = new BasicConv2d(
            maxpool1->getOutputDims(), 64, 80, {1, 1}, {1, 1}, {0, 0},
            "conv2d_3b_1x1"
        );
        addLayer("", conv2d_3b_1x1);
        conv2d_4a_3x3 = new BasicConv2d(
            conv2d_3b_1x1->getOutputDims(), 80, 192, {3, 3}, {1, 1}, {0, 0},
            "conv2d_4a_3x3"
        );
        addLayer("", conv2d_4a_3x3);

        maxpool2 = new CUDANet::Layers::MaxPooling2d(
            conv2d_4a_3x3->getOutputDims(), 192, {3, 3}, {2, 2}, {0, 0},
            CUDANet::Layers::ActivationType::NONE
        );
        addLayer("maxpool2", maxpool2);

        Mixed_5b =
            new InceptionA(maxpool2->getOutputDims(), 192, 32, "Mixed_5b");
        addLayer("", Mixed_5b);
        Mixed_5c =
            new InceptionA(Mixed_5b->getOutputDims(), 256, 64, "Mixed_5c");
        addLayer("", Mixed_5c);
        Mixed_5d =
            new InceptionA(Mixed_5c->getOutputDims(), 288, 64, "Mixed_5d");
        addLayer("", Mixed_5d);

        Mixed_6a = new InceptionB(Mixed_5d->getOutputDims(), 288, "Mixed_6a");
        addLayer("", Mixed_6a);

        Mixed_6b =
            new InceptionC(Mixed_6a->getOutputDims(), 768, 128, "Mixed_6b");
        addLayer("", Mixed_6b);
        Mixed_6c =
            new InceptionC(Mixed_6b->getOutputDims(), 768, 160, "Mixed_6c");
        addLayer("", Mixed_6c);
        Mixed_6d =
            new InceptionC(Mixed_6c->getOutputDims(), 768, 160, "Mixed_6d");
        addLayer("", Mixed_6d);
        Mixed_6e =
            new InceptionC(Mixed_6d->getOutputDims(), 768, 192, "Mixed_6e");
        addLayer("", Mixed_6e);

        Mixed_7a = new InceptionD(Mixed_6e->getOutputDims(), 768, "Mixed_7a");
        addLayer("", Mixed_7a);

        Mixed_7b = new InceptionE(Mixed_7a->getOutputDims(), 1280, "Mixed_7b");
        addLayer("", Mixed_7b);
        Mixed_7c = new InceptionE(Mixed_7b->getOutputDims(), 2048, "Mixed_7c");
        addLayer("", Mixed_7c);

        fc = new CUDANet::Layers::Dense(
            Mixed_7c->getOutputSize(), 1000,
            CUDANet::Layers::ActivationType::SOFTMAX
        );
        addLayer("", fc);
    }

    float* predict(const float* input) {
        float *d_x = inputLayer->forward(input);
        
        d_x = conv2d_1a_3x3->forward(d_x);
        d_x = conv2d_2a_3x3->forward(d_x);
        d_x = conv2d_2b_3x3->forward(d_x);
        d_x = maxpool1->forward(d_x);
        d_x = conv2d_3b_1x1->forward(d_x);
        d_x = conv2d_4a_3x3->forward(d_x);
        d_x = maxpool2->forward(d_x);
        d_x = Mixed_5b->forward(d_x);
        d_x = Mixed_5c->forward(d_x);
        d_x = Mixed_5d->forward(d_x);
        d_x = Mixed_6a->forward(d_x);
        d_x = Mixed_6b->forward(d_x);
        d_x = Mixed_6c->forward(d_x);
        d_x = Mixed_6d->forward(d_x);
        d_x = Mixed_6e->forward(d_x);
        d_x = Mixed_7a->forward(d_x);
        d_x = Mixed_7b->forward(d_x);
        d_x = Mixed_7c->forward(d_x);
        d_x = fc->forward(d_x);

        float* output = outputLayer->forward(d_x);
        return output;
    }

    ~InceptionV3() {
        delete conv2d_1a_3x3;
        delete conv2d_2a_3x3;
        delete conv2d_2b_3x3;
        delete maxpool1;
        delete conv2d_3b_1x1;
        delete conv2d_4a_3x3;
        delete maxpool2;
        delete Mixed_5b;
        delete Mixed_5c;
        delete Mixed_5d;
        delete Mixed_6a;
        delete Mixed_6b;
        delete Mixed_6c;
        delete Mixed_6d;
        delete Mixed_6e;
        delete Mixed_7a;
        delete Mixed_7b;
        delete Mixed_7c;
        delete fc;
    }


  private:
    BasicConv2d *conv2d_1a_3x3;
    BasicConv2d *conv2d_2a_3x3;
    BasicConv2d *conv2d_2b_3x3;

    CUDANet::Layers::MaxPooling2d *maxpool1;

    BasicConv2d *conv2d_3b_1x1;
    BasicConv2d *conv2d_4a_3x3;

    CUDANet::Layers::MaxPooling2d *maxpool2;

    InceptionA *Mixed_5b;
    InceptionA *Mixed_5c;
    InceptionA *Mixed_5d;

    InceptionB *Mixed_6a;

    InceptionC *Mixed_6b;
    InceptionC *Mixed_6c;
    InceptionC *Mixed_6d;
    InceptionC *Mixed_6e;

    InceptionD *Mixed_7a;

    InceptionE *Mixed_7b;
    InceptionE *Mixed_7c;

    CUDANet::Layers::Dense *fc;
};