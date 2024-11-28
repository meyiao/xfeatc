#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>


class Matcher {
public:

    static void Match(const cv::Mat &descs1, const cv::Mat &descs2, std::vector<cv::DMatch> &matches, float minScore = 0.82f);

    static bool RejectBadMatchesF(std::vector<cv::Point2f> &pts1,
                                  std::vector<cv::Point2f> &pts2,
                                  std::vector<cv::DMatch> &matches,
                                  float thresh);


};