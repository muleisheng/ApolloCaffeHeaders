#pragma once

#include <caffe/blob.hpp>
#include <caffe/layer.hpp>
#include <caffe/proto/caffe.pb.h>

namespace smartseg {

using caffe::Blob;

template <typename Dtype>
class DilationLayer : public caffe::Layer<Dtype> {
public:
    explicit DilationLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
        _param = param.smartseg_dilation_param();
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
        return "SmartsegDilation";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    DilationParameter _param;
};

}