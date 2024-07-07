#include <iostream>
#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"
#include "XFeat.h"


int main(int argc, char** argv) {
    // parse arguments
    const std::string argKeys =
            "{model | ../model/xfeat.onnx | model file path}"
            "{img | ../data/1.png | image file path}";
    cv::CommandLineParser parser(argc, argv, argKeys);
    auto modelFile = parser.get<std::string>("model");
    auto imgFile = parser.get<std::string>("img");
    std::cout << "model file: " << modelFile << std::endl;
    std::cout << "image file: " << imgFile << std::endl;

    // create XFeat object
    std::cout << "creating XFeat...\n";
    XFeat xfeat(modelFile);

    // read image
    std::cout << "reading image...\n";
    cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);

    // detect xfeat corners and compute descriptors
    std::vector<cv::KeyPoint> keys;
    cv::Mat descs;
    xfeat.DetectAndCompute(img, keys, descs, 1000);

    // draw keypoints
    cv::Mat imgColor;
    cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
    cv::drawKeypoints(imgColor, keys, imgColor, cv::Scalar(0, 0, 255));
    cv::putText(imgColor, "features: " + std::to_string(keys.size()), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    cv::imshow("image", imgColor);
    cv::waitKey(0);

    return 0;
}