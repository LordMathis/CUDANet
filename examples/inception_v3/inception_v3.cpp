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
        const int inputSize,
        const int inputChannels,
        const int outputChannels,
        const int kernelSize,
        const int stride,
        const int padding,
        const std::string& prefix
    )
        : inputSize(inputSize),
          inputChannels(inputChannels),
          outputChannels(outputChannels) {
        // Create the convolution layer
        CUDANet::Layers::Conv2d *conv = new CUDANet::Layers::Conv2d(
            inputSize, inputChannels, kernelSize, stride, outputChannels, padding, CUDANet::Layers::ActivationType::NONE
        );

        int batchNormSize = conv->getOutputSize();

        CUDANet::Layers::BatchNorm *batchNorm =
            new CUDANet::Layers::BatchNorm(
                batchNormSize, outputChannels, 1e-3f, CUDANet::Layers::ActivationType::RELU
            );

        addLayer(prefix + ".conv", conv);
        addLayer(prefix + ".bn", batchNorm);
    }

  private:
    int inputSize;
    int inputChannels;
    int outputChannels;
};
