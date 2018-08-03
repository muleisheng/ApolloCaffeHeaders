#pragma once

namespace smartseg {

template <typename Dtype>
class ConfigLayer : public caffe::Layer<Dtype> {
public:
    explicit ConfigLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    bool ShareInParallel() const override {
        return true;
    }
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override {
    }
    int ExactNumBottomBlobs() const override {
        return 0;
    }
    int ExactNumTopBlobs() const override {
        return 0;
    }
    const char* type() const override {
        return "SmartsegConfig";
    }

protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override {
    }
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override {
    }
};

}