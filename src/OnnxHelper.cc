#include "OnnxHelper.h"


void OnnxHelper::GetModelInfo(Ort::Session &session, std::vector<TensorInfo> &inputInfos,
                              std::vector<TensorInfo> &outputInfos) {
    inputInfos.clear();
    outputInfos.clear();
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    std::cout << "Number of inputs = " << numInputNodes << std::endl;
    for (size_t i = 0; i < numInputNodes; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator).get();
        auto inputTypeInfo = session.GetInputTypeInfo(i);
        std::vector<int64_t> inShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        inputInfos.emplace_back(TensorInfo{name, inShape, inputTypeInfo.GetONNXType()});
    }

    size_t numOutputNodes = session.GetOutputCount();
    std::cout << "Number of outputs = " << numOutputNodes << std::endl;
    for (size_t i = 0; i < numOutputNodes; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator).get();
        auto outputTypeInfo = session.GetOutputTypeInfo(i);
        std::vector<int64_t> outShape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        outputInfos.emplace_back(TensorInfo{name, outShape, outputTypeInfo.GetONNXType()});
    }

}


void OnnxHelper::PrintModelInfo(const std::vector<TensorInfo> &inputInfos, const std::vector<TensorInfo> &outputInfos) {
    std::cout << "Input Infos:" << std::endl;
    for (const auto &inputInfo : inputInfos) {
        std::cout << "Name: " << inputInfo.name << ", Shape: [";
        for (int i = 0; i < 4; i++) {
            std::cout << inputInfo.shape[i] << ", ";
        }
        std::cout << "], Type: " << inputInfo.type << std::endl;
    }

    std::cout << "Output Infos:" << std::endl;
    for (const auto &outputInfo : outputInfos) {
        std::cout << "Name: " << outputInfo.name << ", Shape: [";
        for (int i = 0; i < 4; i++) {
            std::cout << outputInfo.shape[i] << ", ";
        }
        std::cout << "], Type: " << outputInfo.type << std::endl;
    }
}



void OnnxHelper::Str2Ort(const std::string& modelFilePath, std::vector<ORTCHAR_T>& modelFileOrt) {
    modelFileOrt.reserve(modelFilePath.size() + 1);
    modelFileOrt.assign(modelFilePath.begin(), modelFilePath.end());
    modelFileOrt.push_back(ORTCHAR_T('\0'));
}