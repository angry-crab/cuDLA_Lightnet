#include "cudla_lightnet.hpp"

namespace cudla_lightnet
{

template <typename T>
T CLAMP(const T& value, const T& low, const T& high)
{
return value < low ? low : (value > high ? high : value);
}

static void convert_float_to_half(float * a, __half * b, int size) {
    for(int i=0; i<size; ++i)
    {
        b[i] = __float2half(a[i]);
    }
}

static void convert_half_to_float(__half * a, float * b, int size) {
    for(int i=0; i<size; ++i)
    {
        b[i] = __half2float(a[i]);
    }
}

static void convert_float_to_int8(const float* a, int8_t* b, int size, float scale) {
    for(int idx=0; idx<size; ++idx)
    {
        float v = (a[idx] / scale);
        if(v < -128) v = -128;
        if(v > 127) v = 127;
        b[idx] = (int8_t)v;
    }
}

std::size_t roundup(std::size_t n, int byte)
{
    int factor = 64 / byte;
    std::size_t res = ((n+factor)/factor) * factor;
    return res;
}

static void reformat(std::vector<float>& input, std::vector<float>& output, std::vector<int>& dim_i, int byte)
{
    std::size_t step_i = roundup(dim_i.back(), byte);
    std::size_t step_o = static_cast<std::size_t>(dim_i.back());
    for(std::size_t pi=0, po=0; pi<input.size(); pi+=step_i, po+=step_o)
    {
        std::memcpy((void*)&output[po], (void*)&input[pi], dim_i.back()*sizeof(float));
    }
}

Lightnet::Lightnet(ModelConfig &model_config, InferenceConfig &inference_config, std::string &engine_path, LightnetBackend backend)
{
    const std::string& model_path = model_config.model_path;
    const std::string& precision = inference_config.precision;
    const int num_class = model_config.num_class;    
    const float score_threshold = model_config.score_threshold;
    const float nms_threshold = model_config.nms_threshold;
    const std::vector<int> anchors = model_config.anchors;
    int num_anchor = model_config.num_anchors;
    const double norm_factor = 1.0;
    const size_t max_workspace_size = inference_config.workspace_size;
    src_width_ = -1;
    src_height_ = -1;
    norm_factor_ = norm_factor;
    batch_size_ = 1;
    multitask_ = 0;
    // Initialize class members
    num_class_ = num_class;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    anchors_ = anchors;
    num_anchor_ = num_anchor;

    mEnginePath = engine_path;
    mBackend    = backend;

    checkCudaErrors(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));

    cudaFree(0);

    mCuDLACtx = new cuDLAContextStandalone(engine_path.c_str());

    void *input_buf = nullptr;
    void *output_buf_0;
    void *output_buf_1;
    void *output_buf_2;

    input_buf    = mCuDLACtx->getInputCudaBufferPtr(0);
    output_buf_0 = mCuDLACtx->getOutputCudaBufferPtr(0);
    output_buf_1 = mCuDLACtx->getOutputCudaBufferPtr(1);
    output_buf_2 = mCuDLACtx->getOutputCudaBufferPtr(2);

    mBindingArray.push_back(reinterpret_cast<void *>(input_buf));
    mBindingArray.push_back(output_buf_0);
    mBindingArray.push_back(output_buf_1);
    mBindingArray.push_back(output_buf_2);

    input_dims = mCuDLACtx->getInputTensorDims(0);
    output_dims_0 = mCuDLACtx->getOutputTensorDims(0);
    output_dims_1 = mCuDLACtx->getOutputTensorDims(1);
    output_dims_2 = mCuDLACtx->getOutputTensorDims(2);
}

Lightnet::~Lightnet()
{
    delete mCuDLACtx;
    mCuDLACtx = nullptr;
    cudaStreamDestroy(mStream);
    // float sum = std::accumulate(time_.begin(), time_.end(), 0);
    // std::cout << "avg infer time : " << sum/time_.size() << std::endl;
}

void Lightnet::preprocess(const std::vector<cv::Mat> &images)
{
    // scales_.clear();
    input_h_.clear();
    const float inputH = static_cast<float>(input_dims[2]);
    const float inputW = static_cast<float>(input_dims[3]);

    // Normalize images and convert to blob directly without additional copying.
    float scale = 1 / 255.0;
    const auto nchw_images = cv::dnn::blobFromImages(images, scale, cv::Size(inputW, inputH), cv::Scalar(0.0, 0.0, 0.0), true);   

    // If the data is continuous, we can use it directly. Otherwise, we need to clone it for contiguous memory.
    input_h_ = nchw_images.isContinuous() ? nchw_images.reshape(1, nchw_images.total()) : nchw_images.reshape(1, nchw_images.total()).clone();
    pushImg(input_h_.data(), 1, true);
}

void Lightnet::pushImg(void *imgBuffer, int numImg, bool fromCPU)
{
    int dim = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];

    if (mBackend == LightnetBackend::CUDLA_FP16)
    {
        std::vector<__half> tmp_fp(dim);
        convert_float_to_half((float *)imgBuffer, tmp_fp.data(), dim);
        checkCudaErrors(cudaMemcpy(mCuDLACtx->getInputCudaBufferPtr(0), (void *)tmp_fp.data(), dim * sizeof(__half), cudaMemcpyHostToDevice));
    }
    if (mBackend == LightnetBackend::CUDLA_INT8)
    {
        std::vector<int8_t> tmp_int(dim);
        convert_float_to_int8((float *)imgBuffer, tmp_int.data(), dim, mInputScale);
        checkCudaErrors(cudaMemcpy(mCuDLACtx->getInputCudaBufferPtr(0), (void *)tmp_int.data(), dim * sizeof(int8_t), cudaMemcpyHostToDevice));
    }
}

void Lightnet::infer()
{
    output_h_.clear();

    // TODO: find another way to sync device
    checkCudaErrors(cudaDeviceSynchronize());

    mCuDLACtx->submitDLATask(mStream);
    checkCudaErrors(cudaDeviceSynchronize());

    // iff output format is fp16
    int dim3_0 = output_dims_0[0] * output_dims_0[1] * output_dims_0[2];
    int r_0 = roundup(output_dims_0[3], mByte);
    std::vector<float> fp_0_float(dim3_0 * r_0);
    copyHalf2Float(fp_0_float, 0);
    // std::vector<float> fp_0(dim3_0 * output_dims_0[3], 0);
    // reformat(fp_0_float, fp_0, output_dims_0, mByte);

    // print_dla_addr((half *)mCuDLACtx->getOutputCudaBufferPtr(0), 25, dim_0, mStream);
    // checkCudaErrors(cudaStreamSynchronize(mStream));

    int dim3_1 = output_dims_1[0] * output_dims_1[1] * output_dims_1[2];
    int r_1 = roundup(output_dims_1[3], mByte);
    std::vector<float> fp_1_float(dim3_1 * r_1);
    copyHalf2Float(fp_1_float, 1);
    // std::vector<float> fp_1(dim3_1 * output_dims_1[3], 0);
    // reformat(fp_1_float, fp_1, output_dims_1, mByte);

    // print_dla_addr((half *)mCuDLACtx->getOutputCudaBufferPtr(1), 20, dim_0, mStream);
    // checkCudaErrors(cudaStreamSynchronize(mStream));
    
    int dim3_2 = output_dims_2[0] * output_dims_2[1] * output_dims_2[2];
    int r_2 = roundup(output_dims_2[3], mByte);
    std::vector<float> fp_2_float(dim3_2 * r_2);
    copyHalf2Float(fp_2_float, 2);
    // std::vector<float> fp_2(dim3_2 * output_dims_2[3], 0);
    // reformat(fp_2_float, fp_2, output_dims_2, mByte);
    
    // output_h_.push_back(fp_0);
    // output_h_.push_back(fp_1);
    // output_h_.push_back(fp_2);

    output_h_.push_back(fp_0_float);
    output_h_.push_back(fp_1_float);
    output_h_.push_back(fp_2_float);
}

void Lightnet::copyHalf2Float(std::vector<float>& out_float, int binding_idx)
{
    int dim_0 = out_float.size();
    std::vector<__half> fp_0(dim_0);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(fp_0.data(), mCuDLACtx->getOutputCudaBufferPtr(binding_idx), dim_0 * sizeof(__half), cudaMemcpyDeviceToHost));
    convert_half_to_float((__half *)fp_0.data(), (float *)out_float.data(), dim_0);
}

void Lightnet::makeBbox(const int imageH, const int imageW)
{
    bbox_.clear();
    int inputW = input_dims[3];
    int inputH = input_dims[2];
    // Channel size formula to identify relevant tensor outputs for bounding boxes.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    int detection_count = 0;
    std::vector<std::vector<int>> dims{output_dims_0, output_dims_1, output_dims_2};
    for (int i = 0; i < output_h_.size(); i++) {
        auto dim = dims[i];
        // int gridW = dim[3];
        int gridW = roundup(dim[3], mByte);
        int gridH = dim[2];
        int chan = dim[1];

        if (chan_size == chan)
        { // Filtering out the tensors that match the channel size for detections.
            std::vector<BBoxInfo> b = decodeTensor(0, imageH, imageW, inputH, inputW, &(anchors_[num_anchor_ * (detection_count) * 2]), num_anchor_, output_h_[i].data(), gridW, gridH, dim[3]);
            bbox_.insert(bbox_.end(), b.begin(), b.end());
            detection_count++;
        }
    }
    bbox_ = nonMaximumSuppression(nms_threshold_, bbox_); // Apply NMS and return the filtered bounding boxes.
    //    bbox_ = nmsAllClasses(nms_threshold_, bbox_, num_class_); // Apply NMS and return the filtered bounding boxes.   
}

std::vector<BBoxInfo> Lightnet::decodeTensor(const int imageIdx, const int imageH, const int imageW,  const int inputH, const int inputW, const int *anchor, const int anchor_num, const float *output, const int gridW, const int gridH, const int gridW_unpad)
{
    const int volume = gridW * gridH;
    // ??????
    const float* detections = &output[imageIdx * volume * anchor_num * (5 + num_class_)];

    std::vector<BBoxInfo> binfo;
    const float scale_x_y = 2.0; // Scale factor used for bounding box center adjustments.
    const float offset = 0.5 * (scale_x_y - 1.0); // Offset for scaled centers.

    for (int y = 0; y < gridH; ++y)
    {
        for (int x = 0; x < gridW; ++x)
        {
            for (int b = 0; b < anchor_num; ++b)
            {
                const int numGridCells = gridH * gridW;
                const int bbindex = (y * gridW + x) + numGridCells * b * (5 + num_class_);

                const float objectness = detections[bbindex + 4 * numGridCells]; // Objectness score.
                if (objectness < score_threshold_)
                {
                    continue; // Skip detection if below threshold.
                }

                // Extract anchor dimensions.
                const float pw = static_cast<float>(anchor[b * 2]);
                const float ph = static_cast<float>(anchor[b * 2 + 1]);

                // Decode bounding box center and dimensions.
                // Scaled-YOLOv4 format for simple and fast decorder
                // bx = tx * 2 + cx - 0.5
                // by = ty * 2 + cy - 0.5
                // bw = (tw * 2) * (tw * 2) * pw
                // bh = (th * 2) * (th * 2) * pw
                // The sigmoid is included in the last layer of the DNN models.
                // Cite in https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982 (Loss for YOLOv3, YOLOv4 and Scaled-YOLOv4)	  
                const float bx = x + scale_x_y * detections[bbindex] - offset;
                const float by = y + scale_x_y * detections[bbindex + numGridCells] - offset;
                const float bw = pw * std::pow(detections[bbindex + 2 * numGridCells] * 2, 2);
                const float bh = ph * std::pow(detections[bbindex + 3 * numGridCells] * 2, 2);

                // Decode class probabilities.
                float maxProb = 0.0f;
                int maxIndex = -1;
                for (int i = 0; i < num_class_; ++i)
                {
                    float prob = detections[bbindex + (5 + i) * numGridCells];
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb *= objectness; // Adjust probability with objectness score.

                // Add bounding box proposal if above the threshold.
                if (maxProb > score_threshold_)
                {
                    const uint32_t strideH = inputH / gridH;
                    const uint32_t strideW = inputW / gridW_unpad;
                    addBboxProposal(bx, by, bw, bh, strideH, strideW, maxIndex, maxProb, imageW, imageH, inputW, inputH, binfo);
                }
            }
        }
    }
    return binfo;

}

void Lightnet::addBboxProposal(const float bx, const float by, const float bw, const float bh,
    const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb,
    const uint32_t image_w, const uint32_t image_h,
    const uint32_t input_w, const uint32_t input_h, std::vector<BBoxInfo>& binfo)
{
    BBoxInfo bbi;
    // Convert the bounding box to the original image scale
    bbi.box = convertBboxRes(bx, by, bw, bh, stride_h_, stride_w_, input_w, input_h);

    // Skip invalid boxes
    if (bbi.box.x1 > bbi.box.x2 || bbi.box.y1 > bbi.box.y2) {
      return;
    }

    // Scale box coordinates to match the size of the original image
    bbi.box.x1 = (bbi.box.x1 / input_w) * image_w;
    bbi.box.y1 = (bbi.box.y1 / input_h) * image_h;
    bbi.box.x2 = (bbi.box.x2 / input_w) * image_w;
    bbi.box.y2 = (bbi.box.y2 / input_h) * image_h;

    // Set label and probability
    bbi.label = maxIndex;
    bbi.prob = maxProb;
    bbi.classId = maxIndex; // Note: 'label' and 'classId' are set to the same value. Consider if this redundancy is necessary.

    // Add the box info to the vector
    binfo.push_back(bbi);
}

BBox Lightnet::convertBboxRes(const float& bx, const float& by, const float& bw, const float& bh,
    const uint32_t& stride_h_, const uint32_t& stride_w_,
    const uint32_t& netW, const uint32_t& netH)
{
    BBox b;
    // Convert coordinates from feature map scale back to original image scale
    float x = bx * stride_w_;
    float y = by * stride_h_;

    // Calculate top-left and bottom-right coordinates
    b.x1 = x - bw / 2;
    b.x2 = x + bw / 2;
    b.y1 = y - bh / 2;
    b.y2 = y + bh / 2;

    // Clamp coordinates to be within the network input dimensions
    b.x1 = CLAMP(b.x1, 0.0f, static_cast<float>(netW));
    b.x2 = CLAMP(b.x2, 0.0f, static_cast<float>(netW));
    b.y1 = CLAMP(b.y1, 0.0f, static_cast<float>(netH));
    b.y2 = CLAMP(b.y2, 0.0f, static_cast<float>(netH));

    return b;
}

std::vector<BBoxInfo> Lightnet::nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
      if (x1min > x2min)
      {
        std::swap(x1min, x2min);
        std::swap(x1max, x2max);
      }
      return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float {
      float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
      float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
      float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
      float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
      float overlap2D = overlapX * overlapY;
      float u = area1 + area2 - overlap2D;
      return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
		     [](const BBoxInfo& b1, const BBoxInfo& b2) { return b1.prob > b2.prob; });
    std::vector<BBoxInfo> out;
    for (auto& i : binfo)
    {
        bool keep = true;
        for (auto& j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i.box, j.box);
                keep = overlap <= nmsThresh;
            } else
            {
                break;
            }
        }
        if (keep) 
        out.push_back(i);
    }
    return out;
}

void Lightnet::makeMask(std::vector<cv::Vec3b> &argmax2bgr)
{
    masks_.clear();
    // Formula to identify output tensors not related to bounding box detections.
    // int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // for (int i = 1; i < trt_common_->getNbBindings(); i++)
    // {
    //     const auto dims = trt_common_->getBindingDimensions(i);
    //     int outputW = dims.d[3];
    //     int outputH = dims.d[2];
    //     int chan = dims.d[1];
    //     // Identifying tensors by channel size and name for segmentation masks.      
    //     if (chan_size != chan)
    //     {
    //         std::string name = trt_common_->getIOTensorName(i);
    //         if (contain(name, "argmax"))
    //         { // Check if tensor name contains "argmax".
    //             cv::Mat mask = cv::Mat::zeros(outputH, outputW, CV_8UC3);
    //             int *buf = (int *)output_h_.at(i-1).get();

    //             for (int y = 0; y < outputH; y++)
    //             {
    //                 int stride = outputW * y;
    //                 cv::Vec3b *ptr = mask.ptr<cv::Vec3b>(y);

    //                 for (int x = 0; x < outputW; x++)
    //                 {
    //                     int id = buf[stride + x];
    //                     ptr[x] = argmax2bgr[id]; // Mapping class index to color.
    //                 }
    //             }
    //             masks_.push_back(mask);
    //         }
    //     }
    // }
}

void Lightnet::makeDepthmap()
{
    depthmaps_.clear();
    // Formula to identify output tensors not related to bounding box detections.
    // int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // for (int i = 1; i < trt_common_->getNbBindings(); i++)
    // {
    //     const auto dims = trt_common_->getBindingDimensions(i);
    //     int outputW = dims.d[3];
    //     int outputH = dims.d[2];
    //     int chan = dims.d[1];
    //     // Identifying tensors by channel size and name for depthmap.      
    //     if (chan_size != chan)
    //     {
    //         std::string name = trt_common_->getIOTensorName(i);
    //         nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);
    //         if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT)
    //         { // Check if tensor name contains "lgx" and tensor type is 'kFLOAT'.
    //             cv::Mat depthmap = cv::Mat::zeros(outputH, outputW, CV_8UC1);
    //             float *buf = (float *)output_h_.at(i-1).get();
    //             for (int y = 0; y < outputH; y++)
    //             {
    //                 int stride = outputW * y;

    //                 for (int x = 0; x < outputW; x++)
    //                 {
    //                     float rel = 1.0 - buf[stride + x];
    //                     int value = (int)(rel * 255);
    //                     depthmap.at<unsigned char>(y, x) = value;
    //                 }
    //             }
    //             depthmaps_.push_back(depthmap);
    //         }
    //     }
    // }
}

std::vector<BBoxInfo> Lightnet::getBbox()
{
    return bbox_;
}

std::vector<cv::Mat> Lightnet::getDepthmap()
{
    return depthmaps_;
}

std::vector<cv::Mat> Lightnet::getMask()
{
    return masks_;
}

void Lightnet::clearSubnetBbox()
{
    subnet_bbox_.clear();
}

void Lightnet::appendSubnetBbox(std::vector<BBoxInfo> bb)    
{
    subnet_bbox_.insert(subnet_bbox_.end(), bb.begin(), bb.end());
}

std::vector<BBoxInfo> Lightnet::getSubnetBbox()
{
    return subnet_bbox_;
}

void Lightnet::drawBbox(cv::Mat &img, std::vector<BBoxInfo> bboxes, std::vector<std::vector<int>> &colormap, std::vector<std::string> names)
{
    for (const auto& bbi : bboxes)
    {
        int id = bbi.classId;
        std::stringstream stream;

        if (!names.empty()) 
        {
            stream << std::fixed << std::setprecision(2) << names[id] << "  " << bbi.prob;
        } else 
        {
            stream << std::fixed << std::setprecision(2) << "id:" << id << "  score:" << bbi.prob;
        }

        cv::Scalar color = colormap.empty() ? cv::Scalar(255, 0, 0) : cv::Scalar(colormap[id][2], colormap[id][1], colormap[id][0]);
        cv::rectangle(img, cv::Point(bbi.box.x1, bbi.box.y1), cv::Point(bbi.box.x2, bbi.box.y2), color, 2);
        cv::putText(img, stream.str(), cv::Point(bbi.box.x1, bbi.box.y1 - 5), 0, 0.5, color, 1);
    }
}

}