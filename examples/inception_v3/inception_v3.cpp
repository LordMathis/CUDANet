#include "inception_v3.hpp"

#include <cudanet.cuh>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<float>
readAndNormalizeImage(const std::string &imagePath, int width, int height) {
    // Read the image using OpenCV
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // Resize and normalize the image
    cv::resize(image, image, cv::Size(width, height));
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    // Normalize the image https://pytorch.org/hub/pytorch_vision_alexnet/
    cv::Mat mean(image.size(), CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std(image.size(), CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
    cv::subtract(image, mean, image);
    cv::divide(image, std, image);

    // Convert the 3D image matrix to a 1D array of floats
    std::vector<float> imageData;
    for (int c = 0; c < image.channels(); ++c) {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                imageData.push_back(image.at<cv::Vec3f>(i, j)[c]);
            }
        }
    }

    return imageData;
}

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
        readAndNormalizeImage(imagePath, inputSize.first, inputSize.second);

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