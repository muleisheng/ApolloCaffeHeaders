#ifndef CAFFE_IMAGE_DISTORT_LAYER_HPP_
#define CAFFE_IMAGE_DISTORT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"

namespace caffe {

/**
 * @brief distort img blobs in a batch
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class ImageDistortLayer : public Layer<Dtype> {
    public:
        explicit ImageDistortLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ImageDistort"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, 
                const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, 
                const vector<Blob<Dtype>*>& bottom);

        Dtype pre_bgr_[3];
        Dtype pre_scale_;
        Dtype new_bgr_[3];
        Dtype new_scale_;
        caffe_fcn_data_layer::DistortionParameter distort_param_;
        caffe_fcn_data_layer::NoiseParameter noise_param_;
        bool need2distort_;
        bool need2noise_;

        // bottom size
        unsigned int num_;
        unsigned int channels_;
        unsigned int height_;
        unsigned int width_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DISTORT_LAYER_HPP_
