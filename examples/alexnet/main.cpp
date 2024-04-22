#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <model.hpp>
#include <conv2d.cuh>
#include <max_pooling.cuh>
#include <dense.cuh>

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

    // Block 1
    CUDANet::Layers::Conv2d *conv1 = new CUDANet::Layers::Conv2d(
        inputSize, inputChannels, 11, 4, 64, 2, CUDANet::Layers::ActivationType::RELU
    );
    model->addLayer("features.0", conv1); // Match pytorch naming
    CUDANet::Layers::MaxPooling2D *pool1 = new CUDANet::Layers::MaxPooling2D(
        56, 64, 3, 2, CUDANet::Layers::ActivationType::NONE
    );
    model->addLayer("pool1", pool1);

    // Block 2
    CUDANet::Layers::Conv2d *conv2 = new CUDANet::Layers::Conv2d(
        27, 64, 5, 1, 192, 2, CUDANet::Layers::ActivationType::RELU
    );
    model->addLayer("features.3", conv2);
    CUDANet::Layers::MaxPooling2D *pool2 = new CUDANet::Layers::MaxPooling2D(
        27, 192, 3, 2, CUDANet::Layers::ActivationType::NONE
    );
    model->addLayer("pool2", pool2);

    // Block 3
    CUDANet::Layers::Conv2d *conv3 = new CUDANet::Layers::Conv2d(
        13, 192, 3, 1, 384, 1, CUDANet::Layers::ActivationType::RELU
    );
    model->addLayer("features.6", conv3);

    // Block 4
    CUDANet::Layers::Conv2d *conv4 = new CUDANet::Layers::Conv2d(
        13, 384, 3, 1, 256, 1, CUDANet::Layers::ActivationType::RELU
    );
    model->addLayer("features.8", conv4);

    // Block 5
    CUDANet::Layers::Conv2d *conv5 = new CUDANet::Layers::Conv2d(
        13, 256, 3, 1, 256, 1, CUDANet::Layers::ActivationType::RELU
    );
    model->addLayer("features.10", conv5);
    CUDANet::Layers::MaxPooling2D *pool5 = new CUDANet::Layers::MaxPooling2D(
        13, 256, 3, 2, CUDANet::Layers::ActivationType::NONE
    );
    model->addLayer("pool5", pool5);

    // Classifier
    CUDANet::Layers::Dense *dense1 = new CUDANet::Layers::Dense(
        6 * 6 * 256, 4096, CUDANet::Layers::ActivationType::RELU
    );
    model->addLayer("classifier.1", dense1);

    CUDANet::Layers::Dense *dense2 = new CUDANet::Layers::Dense(
        4096, 4096, CUDANet::Layers::ActivationType::RELU
    );
    model->addLayer("classifier.4", dense2);

    CUDANet::Layers::Dense *dense3 = new CUDANet::Layers::Dense(
        4096, 1000, CUDANet::Layers::ActivationType::NONE
    );
    model->addLayer("classifier.6", dense3);

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

    model->validate();

    model->loadWeights(modelWeightsPath);

    // Read and normalize the image
    std::vector<float> imageData = readAndNormalizeImage(imagePath, inputSize, inputSize);

    // Print the size of the image data
    float* output = model->predict(imageData.data());

    // Get max index
    int maxIndex = 0;
    for (int i = 0; i < outputSize; i++) {
        if (output[i] > output[maxIndex]) {
            maxIndex = i;
        }
    }

    std::cout << "Prediction: " << maxIndex << std::endl;
    return 0;
}