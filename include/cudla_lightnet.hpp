#ifndef CUDLA_LIGHTNET_HPP
#define CUDLA_LIGHTNET_HPP

#include "NvInfer.h"
#include <NvInferPlugin.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

#include "cudla_context_standalone.h"

namespace cudla_lightnet
{

#define checkCudaErrors(call)                                                                                          \
    {                                                                                                                  \
        cudaError_t ret = (call);                                                                                      \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cout << "Cuda failure: " << cudaGetErrorString(ret) << " at line " << __LINE__ << " in file "         \
                      << __FILE__ << " error status: " << ret << std::endl;                                            \
            abort();                                                                                                   \
        }                                                                                                              \
    }

enum LightnetBackend
{
    TRT_GPU    = 1, // NOT USED
    CUDLA_FP16 = 2,
    CUDLA_INT8 = 3,
};

/**
 * Represents a bounding box in a 2D space.
 */
struct BBox
{
  float x1, y1; ///< Top-left corner of the bounding box.
  float x2, y2; ///< Bottom-right corner of the bounding box.
};

/**
 * Contains information about a detected object, including its bounding box,
 * label, class ID, and the probability of the detection.
 */
struct BBoxInfo
{
  BBox box; ///< Bounding box of the detected object.
  int label; ///< Label of the detected object.
  int classId; ///< Class ID of the detected object.
  float prob; ///< Probability of the detection.
};

/**
 * Represents a colormap entry, including an ID, a name, and a color.
 * This is used for mapping class IDs to human-readable names and visual representation colors.
 */
typedef struct Colormap_
{
  int id; ///< ID of the color map entry.
  std::string name; ///< Human-readable name associated with the ID.
  std::vector<unsigned char> color; ///< Color associated with the ID, typically in RGB format.
} Colormap;

class Lightnet
{
public:

  Lightnet(ModelConfig &model_config, InferenceConfig &inference_config, std::string &engine_path, LightnetBackend backend);

  ~Lightnet();

  void infer();

  void preprocess(const std::vector<cv::Mat> &images);

  void pushImg(void *imgBuffer, int numImg, bool fromCPU);

  void infer();

  void copyHalf2Float(std::vector<float>& out_float, int binding_idx);

  void makeBbox(const int imageH, const int imageW);

  std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH, const int imageW,  const int inputH, const int inputW, const int *anchor, const int anchor_num, const float *output, const int gridW, const int gridH);

  void addBboxProposal(const float bx, const float by, const float bw, const float bh,
				    const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb,
				    const uint32_t image_w, const uint32_t image_h,
				    const uint32_t input_w, const uint32_t input_h, std::vector<BBoxInfo>& binfo);
  
  BBox convertBboxRes(const float& bx, const float& by, const float& bw, const float& bh,
    const uint32_t& stride_h_, const uint32_t& stride_w_,
    const uint32_t& netW, const uint32_t& netH);
  
  std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo);

  void makeMask(std::vector<cv::Vec3b> &argmax2bgr);

  void makeDepthmap();

  void drawBbox(cv::Mat &img, std::vector<BBoxInfo> bboxes, std::vector<std::vector<int>> &colormap, std::vector<std::string> names);

  std::vector<BBoxInfo> getBbox();

  std::vector<cv::Mat> getDepthmap();

  std::vector<cv::Mat> getMask();

  cudaStream_t mStream;

  cuDLAContextStandalone *mCuDLACtx;

  std::vector<void *> mBindingArray;

  std::string   mEnginePath;

  LightnetBackend mBackend;

  std::vector<float> input_h_;

  std::vector<float> output_h0_;
  std::vector<float> output_h1_;
  std::vector<float> output_h2_;

  std::vector<int> input_dims{1, 3, 960, 960};
  std::vector<int> output_dims_0{1, 75, 30, 30};
  std::vector<int> output_dims_1{1, 75, 60, 60};
  std::vector<int> output_dims_2{1, 75, 120, 120};

  // Output : 2 for fp16, 1 for in8
  int mByte = 2;

  // mInputScale is obtained from calibration file
  float mInputScale   = 2.00787;

  /**
   * Number of classes that the model is trained to predict.
   */
  int num_class_;

  /**
   * Threshold for filtering out predictions with a confidence score lower than this value.
   */
  float score_threshold_;

  /**
   * Threshold used by the Non-Maximum Suppression (NMS) algorithm to filter out overlapping bounding boxes.
   */
  float nms_threshold_;

  /**
   * List of anchor dimensions used by the model for detecting objects. Anchors are pre-defined sizes and ratios that the model uses as reference points for object detection.
   */
  std::vector<int> anchors_;

  /**
   * Number of anchors used by the model. This typically corresponds to the size of the `anchors_` vector.
   */
  int num_anchor_;

  /**
   * The size of batches processed by the model during inference. A larger batch size can improve throughput but requires more memory.
   */
  int batch_size_;

  /**
   * Normalization factor applied to the input images before passing them to the model. This is used to scale the pixel values to a range the model expects.
   */
  double norm_factor_;

  /**
   * Width of the source images before any preprocessing. This is used to revert any scaling or transformations for visualization or further processing.
   */
  int src_width_;

  /**
   * Height of the source images before any preprocessing. Similar to `src_width_`, used for reverting transformations.
   */
  int src_height_;

  /**
   * Flag indicating whether the model performs multiple tasks beyond object detection, such as segmentation or classification.
   */
  int multitask_;

  /**
   * Stores bounding boxes detected by the primary network.
   */  
  std::vector<BBoxInfo> bbox_;

  /**
   * Stores mask images for each detected object, typically used in segmentation tasks.
   */
  std::vector<cv::Mat> masks_;

  /**
   * Stores depth maps generated from the network's output, providing depth information for each pixel.
   */
  std::vector<cv::Mat> depthmaps_;

  /**
   * Stores bounding boxes detected by the subnet, allowing for specialized processing on selected detections.
   */
  std::vector<BBoxInfo> subnet_bbox_;

};

}

#endif