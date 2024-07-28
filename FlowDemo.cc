#include <iostream>
#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"
#include "XFeat.h"
#include "Matcher.h"


int main(int argc, char** argv) {
    // parse arguments
    const std::string argKeys =
            "{dir |  | data directory path}"
            "{model | ../model/xfeat.onnx | model file path}";
    cv::CommandLineParser parser(argc, argv, argKeys);
    auto modelFile = parser.get<std::string>("model");
    auto dir = parser.get<std::string>("dir");
    std::cout << "model file: " << modelFile << std::endl;
    std::cout << "directory: " << dir << std::endl;

    // create XFeat object
    std::cout << "creating XFeat...\n";
    XFeat xfeat(modelFile);


    std::vector<cv::KeyPoint> keys1;
    cv::Mat descs1;
    cv::Mat img1;

    // detect xfeat corners and compute descriptors
    for (int n = 0; n < 1000; n+=1) {
        std::string imgFile = dir + "/" + std::to_string(n) + ".png";
        cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(640, 640));

        std::vector<cv::KeyPoint> keys2;
        cv::Mat descs2;
        xfeat.DetectAndCompute(img, keys2, descs2, 1000);

        if (keys1.size() > 0) {
            // matching
            std::vector<cv::DMatch> matches;
            Matcher::Match(descs1, descs2, matches, 0.82f);

            // draw matches
            cv::Mat imgMatches;
            cv::cvtColor(img, imgMatches, cv::COLOR_GRAY2BGR);
            for (const auto &m : matches) {
                cv::line(imgMatches, keys1[m.queryIdx].pt, keys2[m.trainIdx].pt, cv::Scalar(0, 255, 0));
                cv::circle(imgMatches, keys2[m.trainIdx].pt, 2, cv::Scalar(0, 0, 255), 2);
            }


//            cv::drawMatches(img1, keys1, img, keys2, matches, imgMatches);
            cv::putText(imgMatches, "matches: " + std::to_string(matches.size()), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);

            cv::imshow("matches", imgMatches);
            cv::waitKey(1);
        }

        keys1 = keys2;
        descs1 = descs2.clone();
        img1 = img.clone();

    }


    return 0;
}