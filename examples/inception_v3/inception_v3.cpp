#include "inception_v3.hpp"

#include <cudanet.cuh>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>


int main(int argc, const char *const argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << "<model_weights_path> <image_path>"
                  << std::endl;
        return 1;  // Return error code indicating incorrect usage
    }

    std::string modelWeightsPath = argv[1];
    std::string imagePath        = argv[2];

    const shape2d inputSize     = {299, 299};
    const int     inputChannels = 3;
    const int     outputSize    = 1000;

    InceptionV3 *inception_v3 =
        new InceptionV3(inputSize, inputChannels, outputSize);
    // inception_v3->printSummary();

    std::cout << std::endl;

    inception_v3->loadWeights(modelWeightsPath);

    std::vector<float> imageData =
        readAndNormalizeImage(imagePath, inputSize.first, inputSize.first);

    // Print the size of the image data
    const float *output = inception_v3->predict(imageData.data());

        // Get max index
    int maxIndex = 0;
    for (int i = 0; i < outputSize; i++) {
        if (output[i] > output[maxIndex]) {
            maxIndex = i;
        }
    }

    std::string classLabel = CUDANet::Utils::IMAGENET_CLASS_MAP.at(maxIndex);

    std::cout << "Prediction: " << maxIndex << " " << classLabel << std::endl;
    return 0;
}