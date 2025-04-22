#ifndef DYNAMIC_OBJECT_MASK_H
#define DYNAMIC_OBJECT_MASK_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <vector>
#include <utility>

using namespace std;


class DynamicObjectMasker {
public:
    DynamicObjectMasker(const string& model_path, float conf_thresh=0.5, float nms_thresh=0.4);
    void setFrame(const cv::Mat& img);
    vector<cv::Rect2i> getDynamicArea();
    torch::Tensor preprocess();
    torch::Tensor predict();
    void postprocess(const torch::Tensor& output);
    vector<torch::Tensor> nms(torch::Tensor preds, float score_thresh, float iou_thresh);
    void clearDynamic();

private:
    torch::jit::script::Module model_;
    float conf_threshold_;
    float nms_threshold_;
    int input_width_ = 640;
    int input_height_ = 384;

    cv::Mat frame_;
    vector<cv::Rect2i> dynamicArea;
    
};



#endif //DYNAMIC_OBJECT_MASK_H
