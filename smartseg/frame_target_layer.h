#pragma once

namespace smartseg {

template <typename Dtype>
class FrameTargetLayer : public caffe::Layer<Dtype> {
public:
    explicit FrameTargetLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
        _param = param.smartseg_frame_target_param();
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    int ExactNumBottomBlobs() const override {
        return 4;
    }
    int ExactNumTopBlobs() const override {
        return 12;
    }
    const char* type() const override {
        return "SmartsegFrameTarget";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    FrameTargetParameter _param;
    FrameTarget _frame_target;
};

}