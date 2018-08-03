#pragma once

namespace smartseg {

template <typename Dtype>
class SideViewExtractorLayer : public BaseExtractorLayer<Dtype> {
public:
    struct Features {
        Dtype* nonempty;
        Dtype* top_height;
    };
    enum { num_features = sizeof(Features) / sizeof(Dtype*) };

    explicit SideViewExtractorLayer(const caffe::LayerParameter& param)
        : BaseExtractorLayer<Dtype>(param) {
        _param = param.smartseg_side_view_extractor_param();
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    int ExactNumBottomBlobs() const override {
        return 2;
    }
    int ExactNumTopBlobs() const override {
        return 1;
    }
    const char* type() const override {
        return "SmartsegSideViewExtractor";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    SideViewExtractorParameter _param;
    Frame* _frame;
    SideView* _view;
};

}