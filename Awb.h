//
// Created by jeremy on 7/12/23.
//

#ifndef DEEP_WHITE_BALANCE_AWB_H
#define DEEP_WHITE_BALANCE_AWB_H
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "XTensorHelper.h"

class Awb {
public:
    explicit Awb(std::string);
    ~Awb();

    cv::Mat predict(cv::Mat);
private:
    f00arr kernelP(f00arr);

    std::tuple<f00arr,f00arr,f00arr>
    getMappingFunc(f00arr, f00arr);

    cv::Mat applyMappingFunc(f00arr, f00arr, f00arr, f00arr);
private:
    Ort::Experimental::Session *mWBSess;
    Ort::SessionOptions mWBSessOp;
    Ort::Env mWBSessEnv{ORT_LOGGING_LEVEL_WARNING, ""};
};


#endif //DEEP_WHITE_BALANCE_AWB_H
