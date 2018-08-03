#pragma once

#include <caffe/blob.hpp>
#include <caffe/layer.hpp>
#include <caffe/proto/caffe.pb.h>

namespace smartseg {

using caffe::Blob;

template <typename Dtype>
class PixelwiseReductionLayer : public caffe::Layer<Dtype> {
public:
    explicit PixelwiseReductionLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
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
        return "SmartsegPixelwiseReduction";
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
};

}