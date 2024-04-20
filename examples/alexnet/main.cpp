#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <model.hpp>
#include <conv2d.cuh>

std::vector<float> readAndNormalizeImage(const std::string& imagePath, int width, int height) {
    // Read the image using OpenCV
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    // Resize and normalize the image
    cv::resize(image, image, cv::Size(width, height));
    image.convertTo(image, CV_32F);
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);

    // Convert the 2D image matrix to a 1D array of floats
    std::vector<float> imageData;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            imageData.push_back(image.at<float>(i, j));
        }
    }

    return imageData;
}

CUDANet::Model* createModel(const int inputSize, const int inputChannels, const int outputSize) {
    CUDANet::Model *model =
        new CUDANet::Model(inputSize, inputChannels, outputSize);

    // AlexNet
    CUDANet::Layers::Conv2d *conv1 = new CUDANet::Layers::Conv2d(
        inputSize, inputChannels, 11, 4, 96, CUDANet::Layers::Padding::SAME, CUDANet::Layers::ActivationType::RELU
    );
    model->addLayer("conv1", conv1);
    CUDANet::Layers::MaxPooling *pool1 = new CUDANet::Layers::MaxPooling(
        3, 2
    )


    return model;
}

int main(int argc, const char* const argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << "<model_weights_path> <image_path>" << std::endl;
        return 1; // Return error code indicating incorrect usage
    }

    // Path to the image file
    std::string modelWeightsPath = argv[1];
    std::string imagePath = argv[2];

    const int inputSize = 227;
    const int inputChannels = 3;
    const int outputSize = 1000;

    CUDANet::Model *model = createModel(inputSize, inputChannels, outputSize);


    // Read and normalize the image
    std::vector<float> imageData = readAndNormalizeImage(imagePath, inputSize, inputSize);

    // Print the size of the image data
    std::cout << "Size of image data: " << imageData.size() << std::endl;

    return 0;
}