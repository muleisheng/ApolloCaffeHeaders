#ifndef CAFFE_CNNSEG_POINT_CLOUD_FEATURE_LAYER_HPP
#define CAFFE_CNNSEG_POINT_CLOUD_FEATURE_LAYER_HPP

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {
namespace cnnseg {

/**
 * @brief Provides data to the Net from point cloud
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template<typename Dtype>
class PointCloudFeatureLayer : public Layer<Dtype> {
public:
    explicit PointCloudFeatureLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "PointCloudFeature"; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int MaxBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

    void init_default_feature(const BirdviewParameter &bird_view_param);

    void birdview_project(const cv::Mat &pc_mat,
                          const BirdviewParameter &bird_view_param,
                          Dtype *transform_data);

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
#ifdef CPU_ONLY
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
#else
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top) {
        Forward_cpu(bottom, top);
    }
#endif

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom) {}
#ifdef CPU_ONLY
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
#else
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom) {}
#endif

    shared_ptr<Caffe::RNG> prefetch_rng_;

    cv::Size img_size_;
    cv::Point2f center_pt_;
    cv::Vec2f angle_res_;
    int pad_size_;
    int view_type_;
    int width_;
    int height_;
    int data_channel_;
    bool use_dense_feat_;
    vector<float> direction_data_;
    vector<float> distance_data_;
};

}  // namespace cnnseg
}  // namespace caffe

#endif  // CAFFE_CNNSEG_POINT_CLOUD_FEATURE_LAYER_HPP
