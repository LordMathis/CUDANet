#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<float>
readAndNormalizeImage(const std::string &imagePath, int resizeSize, int cropSize) {
    // Read the image using OpenCV
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    // Convert the image from BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Calculate the scaling factor
    double scale = std::max(static_cast<double>(resizeSize) / image.cols, static_cast<double>(resizeSize) / image.rows);

    // Resize the image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_AREA);

    // Calculate the cropping coordinates
    int x = (resized.cols - cropSize) / 2;
    int y = (resized.rows - cropSize) / 2;

    // Perform center cropping
    cv::Rect roi(x, y, cropSize, cropSize);
    image = resized(roi);

    // Normalize the image
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

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