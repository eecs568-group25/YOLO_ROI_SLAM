#include <DynamicObjectMask.h>

DynamicObjectMasker::DynamicObjectMasker(const string &model_path, float conf_thresh, float nms_thresh)
    : conf_threshold_(conf_thresh), nms_threshold_(nms_thresh){
    //torch::jit::getProfilingMode() = false;
    //torch::jit::getExecutorMode() = false;
    torch::jit::setTensorExprFuserEnabled(false);
    model_ = torch::jit::load(model_path);
}

void DynamicObjectMasker::setFrame(const cv::Mat &img){
    frame_ = img.clone();
}

vector<cv::Rect2i> DynamicObjectMasker::getDynamicArea(){
    return dynamicArea;
}

void DynamicObjectMasker::clearDynamic(){
    dynamicArea.clear();
}

torch::Tensor DynamicObjectMasker::preprocess(){
    if (frame_.empty()){
        throw runtime_error("Frame is empty!");
    }
    cv::Mat resized;
    cv::resize(frame_, resized, cv::Size(input_width_, input_height_));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    torch::Tensor tensor = torch::from_blob(resized.data, {input_height_, input_width_, 3}, torch::kByte)
                               .permute({2, 0, 1})
                               .toType(torch::kFloat)
                               .div(255)
                               .unsqueeze(0);
    return tensor;
}

torch::Tensor DynamicObjectMasker::predict(){
    torch::Tensor input = preprocess();
    torch::Tensor output = model_.forward({input}).toTuple()->elements()[0].toTensor();
    return output;
}

void DynamicObjectMasker::postprocess(const torch::Tensor &output){
    vector<torch::Tensor> detections = nms(output, conf_threshold_, nms_threshold_);
    if (detections.size() > 0){
        for (size_t i=0; i < detections[0].sizes()[0]; ++ i){
            float left = detections[0][i][0].item().toFloat() * frame_.cols / 640;
            float top = detections[0][i][1].item().toFloat() * frame_.rows / 384;
            float right = detections[0][i][2].item().toFloat() * frame_.cols / 640;
            float bottom = detections[0][i][3].item().toFloat() * frame_.rows / 384;
            //float score = detections[0][i][4].item().toFloat();
            int classID = detections[0][i][5].item().toInt();
            // 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 14: bird, 15: cat, 16: dog
            if (classID == 0 || classID == 1 || classID == 2 || classID == 3 || classID == 5 
                || classID == 7 || classID == 14 || classID == 15 || classID == 16){
                cv::Rect2i rect(left, top, right - left, bottom - top);
                dynamicArea.push_back(rect);
            }
        }
    }

}

vector<torch::Tensor> DynamicObjectMasker::nms(torch::Tensor preds, float score_thresh, float iou_thresh)
{
    vector<torch::Tensor> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0)
            continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;

            // FIlter by IOUs
            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}
