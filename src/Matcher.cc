//
// Created by d on 2024/7/13.
//

#include "Matcher.h"


void Matcher::Match(const cv::Mat &descs1, const cv::Mat &descs2,
                    std::vector<cv::DMatch> &matches, float minScore) {
    cv::Mat scores12 = descs1 * descs2.t();
    cv::Mat scores21 = descs2 * descs1.t();
    std::vector<int> match12(descs1.rows, -1);
    for (int i = 0; i < scores12.rows; i++) {
        auto *row = scores12.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for (int j = 1; j < scores12.cols; j++) {
            if (row[j] > maxScore) {
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match12[i] = maxIdx;
    }

    std::vector<int> match21(descs2.rows, -1);
    for (int i = 0; i < scores21.rows; i++) {
        auto *row = scores21.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for (int j = 1; j < scores21.cols; j++) {
            if (row[j] > maxScore) {
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match21[i] = maxIdx;
    }

    // cross-check
    matches.clear();
    for (int i = 0; i < descs1.rows; i++) {
        int j = match12[i];
        if (match21[j] == i && scores12.at<float>(i, j) > minScore) {
            matches.emplace_back(i, j, scores12.at<float>(i, j));
        }
    }
}