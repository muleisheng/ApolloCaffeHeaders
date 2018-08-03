#pragma once

namespace smartseg {

template <typename Dtype>
class SideViewLayer : public caffe::Layer<Dtype> {
public:
    explicit SideViewLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
        _param = param.smartseg_side_view_param();
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
        return "SmartsegSideView";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    SideViewParameter _param;
    SideView _view;
    Frame* _frame;
};

}