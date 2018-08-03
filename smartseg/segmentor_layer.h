#pragma once

namespace smartseg {

template <typename Dtype>
class SegmentorLayer : public caffe::Layer<Dtype> {
public:
    explicit SegmentorLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
        _param = param.smartseg_segmentor_param();
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    int ExactNumBottomBlobs() const override {
        return 8;
    }
    /* int MinBottomBlobs() const override {
        return 7;
    }
    int MaxBottomBlobs() const override {
        return 8;
    } */
    int ExactNumTopBlobs() const override {
        return 1;
    }
    const char* type() const override {
        return "SmartsegSegmentor";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    SegmentorParameter _param;
    Segmentor _segmentor;
};

}