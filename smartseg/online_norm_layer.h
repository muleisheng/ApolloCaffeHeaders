#pragma once

#include <caffe/blob.hpp>
#include <caffe/layer.hpp>
#include <caffe/proto/caffe.pb.h>

namespace smartseg {

using caffe::Blob;

template <typename Dtype>
class OnlineNormLayer : public caffe::Layer<Dtype> {
public:
    explicit OnlineNormLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
        _param = param.smartseg_online_norm_param();
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    int ExactNumBottomBlobs() const override {
        return 1;
    }
    int ExactNumTopBlobs() const override {
        return 1;
    }
    const char* type() const override {
        return "SmartsegOnlineNorm";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
    void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    OnlineNormParameter _param;

    Blob<Dtype> _mean, _variance, _temp, _x_norm;
    int _iters;
    bool _use_global_stats;
    Dtype _moving_average_fraction;
    int _channels;
    Dtype _eps;

    Blob<Dtype> _batch_sum_multiplier;
    Blob<Dtype> _num_by_chans;
    Blob<Dtype> _spatial_sum_multiplier;
};

}