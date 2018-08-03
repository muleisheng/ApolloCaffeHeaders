#ifndef CAFFE_PYRAMID_IMAGE_RUIV2_DATA_LAYERS_HPP_
#define CAFFE_PYRAMID_IMAGE_RUIV2_DATA_LAYERS_HPP_
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"

#include "caffe/detect2d/BlockPack.hpp"
#include "caffe/util/io.hpp"//for debug use only

using namespace caffe_fcn_data_layer;
using namespace std;
namespace caffe {

template<typename Dtype>
class PyramidImageRuiDataLayer: public Layer<Dtype> {
public:
    explicit PyramidImageRuiDataLayer(const LayerParameter &param) :
            Layer<Dtype>(param) {
    }
    virtual ~PyramidImageRuiDataLayer() {
    }
    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom, 
                 const vector<Blob<Dtype> *> &top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                 const vector<Blob<Dtype>*>& top) {
    }
    virtual inline const char* type() const {
        return "PyramidImageRuiv2Data";
    }
    virtual inline int ExactNumBottomBlobs() const {
        return 0;
    }
    virtual inline int ExactNumTopBlobs() const {
        return 2;
    }

    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom, 
                 const vector<Blob<Dtype> *> &top);
#ifdef CPU_ONLY
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                 const vector<Blob<Dtype>*>& top) {}
#else
    virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom, 
                 const vector<Blob<Dtype> *> &top);
#endif

    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top, 
                 const vector<bool> &propagate_down,
                 const vector<Blob<Dtype> *> &bottom) {
    }
    virtual void Backward_gpu(const vector<Blob<Dtype> *> &top, 
                 const vector<bool> &propagate_down,
                 const vector<Blob<Dtype> *> &bottom) {
    }

    void FetchOutterImageFrame(cv::Mat &rosImage);

#ifdef CPU_ONLY
    bool read_img_to_blob_gpu(const float* cv_img, Blob<Dtype>& dst, Dtype mean_c1, Dtype mean_c2, 
                              Dtype mean_c3, int num_channels, int width, int height) {
    }
#else
    bool read_img_to_blob_gpu(const float* cv_img, Blob<Dtype>& dst, Dtype mean_c1, Dtype mean_c2,
                              Dtype mean_c3, int num_channels, int width, int height);
#endif
    Dtype img_w_;
    Dtype img_h_;
    Dtype mean_bgr_[3];

    int heat_map_a_;
    int heat_map_b_;

    BlockPack<Dtype> block_packer_1_;
    Blob<Dtype> img_blob_1_;
    Blob<Dtype> buffered_block_1_;
    vector<Dtype> set_pymid_scales_;

protected:
    bool ReadImgToBlob(const cv::Mat& cv_img, Blob<Dtype>& dst, Dtype mean_c1, Dtype mean_c2,
            Dtype mean_c3);
    bool ReadImgToBlobGPU(const cv::Mat& cv_img, Blob<Dtype>& dst, Dtype mean_c1, Dtype mean_c2,
        Dtype mean_c3);

    void ShowImg(const vector<Blob<Dtype> *> &top);
    virtual int SerializeToBlob(Blob<Dtype> &blob, int start = 0);
};

}// namespace caffe

#endif  // CAFFE_PYRAMID_DATA_LAYERS_HPP_

