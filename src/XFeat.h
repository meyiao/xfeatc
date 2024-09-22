#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "OnnxHelper.h"



class XFeat {
public:
    explicit XFeat(const std::string &modelFile);

    void DetectAndCompute(const cv::Mat &img, std::vector<cv::KeyPoint> &keys, cv::Mat &descs, int maxCorners);


    struct ScoredPoint {
        int x;
        int y;
        float score;
    };

    void Nms(const cv::Mat &scores, float scoreThresh, int kernelSize, std::vector<ScoredPoint>& points);

private:
    void ReshapeScore(const float* src, float *dst, int h, int w, int c);

    void SoftmaxScore(float *score, int h, int w, int c);

    void FlattenScore(float *src, float *dst);

private:
    std::unique_ptr<Ort::Env> ortEnv_;
    std::unique_ptr<Ort::Session> ortSession_;

    // input and output infos
    std::vector<TensorInfo> inputInfos_;
    std::vector<TensorInfo> outputInfos_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;

    // input image height and width
    int H_;
    int W_;
    int Hd8_;   // H/8
    int Wd8_;   // W/8

    // for nms
    const int nmsKernelSize_ = 5;
    std::vector<ScoredPoint> scoredPoints_;

};