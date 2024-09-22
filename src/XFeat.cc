#include "XFeat.h"
#include "OnnxHelper.h"
#include "Timer.h"


inline void CalcBicubicWeights(float t, float &wm1, float &w0, float &w1, float &w2) {
    constexpr float a = -0.75f;
    float t2 = t * t;
    float t3 = t2 * t;
    wm1 = a * (t3 - 2 * t2 + t);
    w0  = (a+2) * t3 - (a+3) * t2 + 1;
    w1  = -(a+2) * t3 + (2*a+3) * t2 - a * t;
    w2  = a * (-t3 + t2);
}

//https://gist.github.com/jrade/293a73f89dfef51da6522428c857802d
inline float FastExp(float x)
{
    constexpr float a = (1 << 23) / 0.69314718f;
    constexpr float b = (1 << 23) * (127 - 0.043677448f);
    x = a * x + b;

    // Remove these lines if bounds checking is not needed
    constexpr float c = (1 << 23);
    constexpr float d = (1 << 23) * 255;
    if (x < c || x > d)
        x = (x < c) ? 0.0f : d;

    // With C++20 one can use std::bit_cast instead
    uint32_t n = static_cast<uint32_t>(x);
    memcpy(&x, &n, 4);
    return x;
}



XFeat::XFeat(const std::string &modelFile) {
    // Convert the modelFile path to onnx compatible path
    std::vector<ORTCHAR_T> modelFileOrt;
    OnnxHelper::Str2Ort(modelFile, modelFileOrt);

    // create onnx runtime session
    ortEnv_ = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "XFeat"));

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    ortSession_ = std::unique_ptr<Ort::Session>(new Ort::Session(*ortEnv_, modelFileOrt.data(), sessionOptions));

    // Get model info
    OnnxHelper::GetModelInfo(*ortSession_, inputInfos_, outputInfos_);

    // set input and output names
    for (const auto &inputInfo : inputInfos_) {
        inputNames_.push_back(inputInfo.name.c_str());
    }
    for (const auto &outputInfo : outputInfos_) {
        outputNames_.push_back(outputInfo.name.c_str());
    }

    H_ = (int)inputInfos_[0].shape[2];
    W_ = (int)inputInfos_[0].shape[3];
    Hd8_ = H_ / 8;
    Wd8_ = W_ / 8;

    // make sure H and W can be divided by 32
    if (H_ % 32 != 0 || W_ % 32 != 0) {
        std::cerr << "Input image size must be divisible by 32!" << std::endl;
        exit(-1);
    }

    // print model info
    OnnxHelper::PrintModelInfo(inputInfos_, outputInfos_);
}



void XFeat::DetectAndCompute(const cv::Mat &img, std::vector<cv::KeyPoint> &keys, cv::Mat &descs, int maxCorners) {
    Timer timer;
    // check image size
    if (img.channels() != inputInfos_[0].shape[1]) {
        std::cerr << "Image channel mismatch!" << img.channels() << std::endl;
        return;
    }

    if (img.rows < H_ || img.cols < W_) {
        std::cerr << "Image size mismatch!" << img.rows << ", " << img.cols << std::endl;
        return;
    }

    const int roiX = (img.cols - W_) / 2;
    const int roiY = (img.rows - H_) / 2;

    // convert image to tensor
    cv::Mat fimg;
    if (img.rows == H_ && img.cols == W_) {
        img.convertTo(fimg, CV_32F, 1.0/255.0);
    } else {
        cv::Rect roi(roiX, roiY, W_, H_);
        cv::Mat roiImg = img(roi);
        roiImg.convertTo(fimg, CV_32F, 1.0/255.0);
    }
    std::vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(OnnxHelper::CreateTensor<float>(inputInfos_[0].shape, fimg.ptr<float>(), W_ * H_, true));

    // run inference
    // outputTensors:
    // 0: [1, 64, H/8, W/8] descriptors
    // 1: [1, 65, H/8, W/8] keypoint scores
    // 2: [1, 1, H/8, W/8] reliability map
    auto outputTensors = ortSession_->Run(Ort::RunOptions{nullptr}, inputNames_.data(),
                                          inputTensors.data(), 1, outputNames_.data(), outputNames_.size());

    // get the keypoint scores, it's a [1, 65, H/8, W/8] tensor,
    auto* kptScorePtr = outputTensors[1].GetTensorMutableData<float>();
    const int shw = Hd8_ * Wd8_;

    // reshape the score into [H/8, W/8, 65] tensor
    timer.Reset();
    std::vector<float> cScore(Hd8_ * Wd8_ * 65);
    Reshape(kptScorePtr, cScore.data(), Hd8_, Wd8_, 65);
    double score_shuffle_time = timer.Elapse();

    timer.Reset();
    // we shall apply softmax along the 65 channels to get the scores
    SoftmaxScore(cScore.data(), Hd8_, Wd8_, 65);
    double score_softmax_time = timer.Elapse();

    timer.Reset();
    // the keypoint score tensor [1, H/8, W/8, 65], we drop the last channel(dust bin), and convert to [H, W] image
    cv::Mat scoreImg(H_, W_, CV_32F);
    auto *scoreImgPtr = scoreImg.ptr<float>();
    FlattenScore(cScore.data(), scoreImgPtr);
    double score_flatten_time = timer.Elapse();

    timer.Reset();
    // apply nms, only keep the points with score > 0.05 and is the local maxima in a 5x5 window
    Nms(scoreImg, 0.05f, nmsKernelSize_, scoredPoints_);
    double nms_time = timer.Elapse();

    timer.Reset();
    // get the reliability map [1, 1, H/8, W/8]
    auto* heatMapPtr = outputTensors[2].GetTensorMutableData<float>();
    cv::Mat heatMapSmall(Hd8_, Wd8_, CV_32F, heatMapPtr);
    // resize it to [H, W]
    cv::Mat heaMapFull;
    cv::resize(heatMapSmall, heaMapFull, cv::Size(W_, H_));
    double heatMap_resize_time = timer.Elapse();

    timer.Reset();
    // multiply the point score with reliability
    for (auto &pt : scoredPoints_) {
        pt.score *= heaMapFull.at<float>(pt.y, pt.x);
    }
    double heatMap_mul_time = timer.Elapse();

    timer.Reset();
    // sort the points by score
    std::sort(scoredPoints_.begin(), scoredPoints_.end(), [](const ScoredPoint &a, const ScoredPoint &b) {
        return a.score > b.score;
    });
    double sort_time = timer.Elapse();

    // convert the scored points to cv::KeyPoint
    // border is calculated in this way:
    // width_scale = Wd8 / (W - 1)
    // pt.x * width_scale - 0.5 > 1 && pt.x * width_scale - 0.5 < width - 2
    const int minEdgeX = 12, maxEdgeX = W_ - 12;
    const int minEdgeY = 12, maxEdgeY = H_ - 12;
    keys.clear();
    for (const auto &pt : scoredPoints_) {
        if (pt.x <= minEdgeX || pt.x >= maxEdgeX || pt.y <= minEdgeY || pt.y >= maxEdgeY) {
            continue;
        }
        keys.emplace_back(pt.x, pt.y, 0);
        if (keys.size() >= maxCorners) {
            break;
        }
    }

    // get the descriptors [1, 64, H/8, W/8]
    auto *descTensorPtr = outputTensors[0].GetTensorMutableData<float>();

    // reshape the descriptors into [H/8, W/8, 64] tensor
    timer.Reset();
    std::vector<float> cDesc(Hd8_ * Wd8_ * 64);
    Reshape(descTensorPtr, cDesc.data(), Hd8_, Wd8_, 64);
    double desc_shuffle_time = timer.Elapse();

    // normalize the descriptors along the channel dimension
    timer.Reset();
    for (int i = 0; i < shw; ++i) {
        double sum = 0;
        float *ptr = &cDesc[i * 64];
        for (int j = 0; j < 64; ++j) {
            sum += ptr[j] * ptr[j];
        }
        float invNorm = static_cast<float>(1.0 / std::max(std::sqrt(sum), 1e-12));
        for (int j = 0; j < 64; ++j) {
            ptr[j] *= invNorm;
        }
    }
    double desc_norm_time = timer.Elapse();

    timer.Reset();
    // bilinear interpolation to get the descriptors
    descs = cv::Mat::zeros((int)keys.size(), 64, CV_32F);
    float wxm1, wx0, wx1, wx2;
    float wym1, wy0, wy1, wy2;
    const float width_scale = float(Wd8_) / float(W_ - 1);
    const float height_scale = float(Hd8_) / float(H_ - 1);
    for (int n = 0; n < (int)(keys.size()); ++n) {
        const auto &pt = keys[n];
        // align_corner = False
        float x = pt.pt.x * width_scale - 0.5f;
        float y = pt.pt.y * height_scale - 0.5f;
//        align_corner = True
//        float x = (pt.pt.x / 639.f * 79.f);
//        float y = (pt.pt.y / 639.f * 79.f);

        // interpolate and normalize the descriptor
        InterpDescriptor(cDesc.data(), descs.ptr<float>(n), x, y);
    }
    double interp_time = timer.Elapse();

    std::cout << "score_shuffle_time=" << score_shuffle_time << ", score_softmax_time=" << score_softmax_time << ", score_flatten_time=" << score_flatten_time << ", nms_time=" << nms_time << std::endl;
    std::cout << "heatmap_resize_time=" << heatMap_resize_time << ", heatmap_mul_time=" << heatMap_mul_time << ", sort_time=" << sort_time << std::endl;
    std::cout << "desc_shuffle_time=" << desc_shuffle_time << ", desc_norm_time=" << desc_norm_time << ", interp_time=" << interp_time << std::endl;
    std::cout << "total_time=" << (score_shuffle_time + score_softmax_time +score_flatten_time + nms_time + heatMap_resize_time + heatMap_mul_time + sort_time + desc_shuffle_time + desc_norm_time + interp_time) << std::endl;

    // add the edge
    for (auto &key : keys) {
        key.pt.x += static_cast<float>(roiX);
        key.pt.y += static_cast<float>(roiY);
    }

}


void XFeat::Nms(const cv::Mat &scores, float scoreThresh, int kernelSize, std::vector<ScoredPoint> &points) {
    points.clear();

    int rows = scores.rows;
    int cols = scores.cols;
    int halfKernelSize = kernelSize / 2;
    cv::Mat mask = cv::Mat::ones(rows, cols, CV_8U);
    const auto * scorePtr = scores.ptr<float>();
    auto* maskPtr = mask.ptr<uchar>();

    std::vector<int> ptrOffsets;
    ptrOffsets.reserve(kernelSize * kernelSize);
    for (int i = -halfKernelSize; i <= halfKernelSize; i++) {
        for (int j = -halfKernelSize; j <= halfKernelSize; j++) {
            if (i == 0 && j == 0) {
                continue;
            }
            ptrOffsets.push_back(i * cols + j);
        }
    }

    for (int i = halfKernelSize; i < rows - halfKernelSize; i++) {
        for (int j = halfKernelSize; j < cols - halfKernelSize; j++) {
            int addr = i * cols + j;
            if (maskPtr[addr] == 0) {
                continue;
            }

            const float score = scorePtr[addr];
            if (score <= scoreThresh) {
                maskPtr[addr] = 0;
                continue;
            }

            // nms
            bool isMax = true;
            for (const auto &offset : ptrOffsets) {
                if (score < scorePtr[addr + offset]) {
                    maskPtr[addr] = 0;
                    isMax = false;
                    break;
                }
            }

            //
            if (isMax) {
                points.push_back({j, i, score});
                // mask out the neighbors
                for (const auto &offset : ptrOffsets) {
                    maskPtr[addr + offset] = 0;
                }
            }
        }
    }


}


void XFeat::Reshape(const float *src, float *dst, int h, int w, int c) {
    // src is in shape [c, h, w], dst is in shape [h, w, c]
    const int hw = h * w;
    int dstIdx = 0;
    for (int i = 0; i < h; i++) {
        const int iw = i * w;
        for (int j = 0; j < w; j++) {
            const int iwj = iw + j;
            for (int k = 0; k < c; k++) {
                dst[dstIdx++] = src[k * hw + iwj];
            }
        }
    }
}


void XFeat::SoftmaxScore(float *score, int h, int w, int c) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float *ptr = score + i * w * c + j * c;
            float sum = 0;
            for (int k = 0; k < c; ++k) {
                float exp = std::exp(ptr[k]);
                ptr[k] = exp;
                sum += exp;
            }
            float invSum = 1.0f / sum;
            for (int k = 0; k < c; ++k) {
                ptr[k] *= invSum;
            }
        }
    }
}


void XFeat::FlattenScore(float *src, float *dst) {
    for (int i = 0; i < Hd8_; ++i) {
        for (int j = 0; j < Wd8_; ++j) {
            float* src_ptr = src + i * Wd8_ * 65 + j * 65;
            int iRow = i * 8;
            int jCol = j * 8;
            float* dst_ptr = dst +iRow * W_ + jCol;
            for (int k = 0; k < 8; ++k) {
                for (int l = 0; l < 8; ++l) {
                    dst_ptr[k * W_ + l] = src_ptr[k * 8 + l];
                }
            }
        }
    }
}


void XFeat::InterpDescriptor(const float *descMat, float *descriptor, float ptx, float pty) {
    int x0 = cvFloor(ptx);
    int y0 = cvFloor(pty);
    int xm1 = x0 - 1;
    int ym1 = y0 - 1;
    float dx = ptx - static_cast<float>(x0);
    float dy = pty - static_cast<float>(y0);

    float wxm1, wx0, wx1, wx2;
    float wym1, wy0, wy1, wy2;
    CalcBicubicWeights(dx, wxm1, wx0, wx1, wx2);
    CalcBicubicWeights(dy, wym1, wy0, wy1, wy2);

    const float* desc_xm1_ym1 = descMat + ym1 * Wd8_ * 64 + xm1 * 64;
    const float* desc_x0_ym1 = desc_xm1_ym1 + 64;
    const float* desc_x1_ym1 = desc_x0_ym1 + 64;
    const float* desc_x2_ym1 = desc_x1_ym1 + 64;
    const float* desc_xm1_y0 = desc_xm1_ym1 + Wd8_ * 64;
    const float* desc_x0_y0 = desc_xm1_y0 + 64;
    const float* desc_x1_y0 = desc_x0_y0 + 64;
    const float* desc_x2_y0 = desc_x1_y0 + 64;
    const float* desc_xm1_y1 = desc_xm1_y0 + Wd8_ * 64;
    const float* desc_x0_y1 = desc_xm1_y1 + 64;
    const float* desc_x1_y1 = desc_x0_y1 + 64;
    const float* desc_x2_y1 = desc_x1_y1 + 64;
    const float* desc_xm1_y2 = desc_xm1_y1 + Wd8_ * 64;
    const float* desc_x0_y2 = desc_xm1_y2 + 64;
    const float* desc_x1_y2 = desc_x0_y2 + 64;
    const float* desc_x2_y2 = desc_x1_y2 + 64;

    double sum = 0;
    for (int i = 0; i < 64; ++i) {
        float v_m1 = wxm1 * desc_xm1_ym1[i] + wx0 * desc_x0_ym1[i] + wx1 * desc_x1_ym1[i] + wx2 * desc_x2_ym1[i];
        float v_0 = wxm1 * desc_xm1_y0[i] + wx0 * desc_x0_y0[i] + wx1 * desc_x1_y0[i] + wx2 * desc_x2_y0[i];
        float v_1 = wxm1 * desc_xm1_y1[i] + wx0 * desc_x0_y1[i] + wx1 * desc_x1_y1[i] + wx2 * desc_x2_y1[i];
        float v_2 = wxm1 * desc_xm1_y2[i] + wx0 * desc_x0_y2[i] + wx1 * desc_x1_y2[i] + wx2 * desc_x2_y2[i];
        float v = wym1 * v_m1 + wy0 * v_0 + wy1 * v_1 + wy2 * v_2;
        descriptor[i] = v;
        sum += v * v;
    }

    // normalize
    float invNorm = static_cast<float>(1.0 / std::max(std::sqrt(sum), 1e-12));
    for (int i = 0; i < 64; ++i) {
        descriptor[i] *= invNorm;
    }
}