#include <iostream>
#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"
#include "XFeat.h"
#include "Matcher.h"


int main(int argc, char** argv) {
    // parse arguments
    const std::string argKeys =
            "{model | ../model/xfeat.onnx | model file path}"
            "{img1 | ../data/1.png | the first image file path}"
            "{img2 | ../data/2.png | the second image file path}";
    cv::CommandLineParser parser(argc, argv, argKeys);
    auto modelFile = parser.get<std::string>("model");
    auto imgFile1 = parser.get<std::string>("img1");
    auto imgFile2 = parser.get<std::string>("img2");
    std::cout << "model file: " << modelFile << std::endl;
    std::cout << "image file 1: " << imgFile1 << std::endl;
    std::cout << "image file 2: " << imgFile2 << std::endl;

    // create XFeat object
    std::cout << "creating XFeat...\n";
    XFeat xfeat(modelFile);

    // read images
    std::cout << "reading images...\n";
    cv::Mat img1 = cv::imread(imgFile1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(imgFile2, cv::IMREAD_GRAYSCALE);

    // detect xfeat corners and compute descriptors
    std::cout << "detecting features ...\n";
    std::vector<cv::KeyPoint> keys1, keys2;
    cv::Mat descs1, descs2;
    xfeat.DetectAndCompute(img1, keys1, descs1, 1000);
    xfeat.DetectAndCompute(img2, keys2, descs2, 1000);

    // draw keypoints
    cv::Mat imgColor1, imgColor2;
    cv::cvtColor(img1, imgColor1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, imgColor2, cv::COLOR_GRAY2BGR);
    cv::drawKeypoints(imgColor1, keys1, imgColor1, cv::Scalar(0, 0, 255));
    cv::drawKeypoints(imgColor2, keys2, imgColor2, cv::Scalar(0, 0, 255));
    cv::putText(imgColor1, "features: " + std::to_string(keys1.size()), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    cv::putText(imgColor2, "features: " + std::to_string(keys2.size()), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    cv::imshow("image1", imgColor1);
    cv::imshow("image2", imgColor2);
    cv::waitKey(0);

    // matching
    std::cout << "matching ...\n";
    std::vector<cv::DMatch> matches;
    Matcher::Match(descs1, descs2, matches, 0.82f);

    // draw matches
    cv::Mat imgMatches;
    cv::drawMatches(imgColor1, keys1, imgColor2, keys2, matches, imgMatches);
    cv::imshow("matches", imgMatches);
    cv::waitKey(0);


    return 0;
}