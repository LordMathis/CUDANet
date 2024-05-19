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
        const int          inputSize,
        const int          inputChannels,
        const int          outputChannels,
        const int          kernelSize,
        const int          stride,
        const int          padding,
        const std::string &prefix
    ) {
        // Create the convolution layer
        CUDANet::Layers::Conv2d *conv = new CUDANet::Layers::Conv2d(
            inputSize, inputChannels, kernelSize, stride, outputChannels,
            padding, CUDANet::Layers::ActivationType::NONE
        );

        int batchNormSize = conv->getOutputSize();

        CUDANet::Layers::BatchNorm *batchNorm = new CUDANet::Layers::BatchNorm(
            batchNormSize, outputChannels, 1e-3f,
            CUDANet::Layers::ActivationType::RELU
        );

        addLayer(prefix + ".conv", conv);
        addLayer(prefix + ".bn", batchNorm);
    }

    float* forward(const float* d_input) {

        for (auto& layer : layers) {
            d_input = layer.second->forward(d_input);
        }
        return d_input;
    }

};

class InceptionA : public CUDANet::Module {
  public:
    InceptionA(
        const int          inputSize,
        const int          inputChannels,
        const int          poolFeatures,
        const std::string &prefix
    )
        : inputSize(inputSize),
          inputChannels(inputChannels),
          poolFeatures(poolFeatures) {
        
        // Branch 1x1
        CUDANet::Module *branch1x1 = new BasicConv2d(
            inputSize, inputChannels, 64, 1, 1, 0, prefix + ".branch1x1"
        );
        addLayer("", branch1x1);

        // Branch 5x5
        CUDANet::Module *branch5x5_1 = new BasicConv2d(
            inputSize, inputChannels, 48, 1, 1, 0, prefix + ".branch5x5_1"
        );
        addLayer("", branch5x5_1);
        CUDANet::Module *branch5x5_2 = new BasicConv2d(
            inputSize, 48, 64, 5, 1, 2, prefix + ".branch5x5_2"
        );
        addLayer("", branch5x5_2);

        // Branch 3x3
        CUDANet::Module *branch3x3_1 = new BasicConv2d(
            inputSize, inputChannels, 64, 1, 1, 0, prefix + ".branch3x3_1"
        );
        addLayer("", branch3x3_1);
        CUDANet::Module *branch3x3_2 = new BasicConv2d(
            inputSize, 64, 96, 3, 1, 1, prefix + ".branch3x3_2"
        );
        addLayer("", branch3x3_2);
        CUDANet::Module *branch3x3_3 = new BasicConv2d(
            inputSize, 96, 96, 3, 1, 1, prefix + ".branch3x3_3"
        );
        addLayer("", branch3x3_3);

        // Branch Pool
        CUDANet::Module *branchPool = new BasicConv2d(
            inputSize, inputChannels, poolFeatures, 1, 1, 0, prefix + ".branchPool"
        );
        addLayer("", branchPool);

        // Concat
        concat_1 = new CUDANet::Layers::Concat(
            branch1x1->getOutputSize(), branch5x5_2->getOutputSize()
        );
        concat_2 = new CUDANet::Layers::Concat(
            concat_1->getOutputSize(), branch3x3_3->getOutputSize()
        );
        
    }

  private:
    int inputSize;
    int inputChannels;
    int poolFeatures;

    CUDANet::Layers::Concat *concat_1;
    CUDANet::Layers::Concat *concat_2;
};