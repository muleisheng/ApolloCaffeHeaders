// -----------------------------------------------------------------------------
// Written by MingLi, liming22@baidu.com, 2017-05-27
// -----------------------------------------------------------------------------

#ifndef CAFFE_DFMB_PSROI_POOLING_LAYER_HPP_
#define CAFFE_DFMB_PSROI_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DFMBPSROIPoolingLayer : public Layer<Dtype> {
    public:
        explicit DFMBPSROIPoolingLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "DFMBPSROIPooling"; }

        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int MaxBottomBlobs() const { return 3; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        Dtype heat_map_a_;
        Dtype heat_map_b_;
        Dtype pad_ratio_;

        int output_dim_;
        bool no_trans_;
        Dtype trans_std_;
        int sample_per_part_;
        int group_height_;
        int group_width_;
        int pooled_height_;
        int pooled_width_;
        int part_height_;
        int part_width_;

        int channels_;
        int height_;
        int width_;

        Blob<Dtype> top_count_;

};

}  // namespace caffe

#endif  // CAFFE_DFMB_PSROI_POOLING_LAYER_HPP_
