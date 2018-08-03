#pragma once

namespace smartseg {

template <typename Dtype>
class HorizontalViewExtractorLayer : public BaseExtractorLayer<Dtype> {
public:
    struct Features {
        Dtype* nonempty;
        Dtype* pitch;
        Dtype* top_distance;
        Dtype* top_inverse_distance;
        Dtype* top_height;
        Dtype* top_intensity;
    };
    enum { num_features = sizeof(Features) / sizeof(Dtype*) };

    explicit HorizontalViewExtractorLayer(const caffe::LayerParameter& param)
        : BaseExtractorLayer<Dtype>(param) {
        _param = param.smartseg_horizontal_view_extractor_param();
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
        return "SmartsegHorizontalViewExtractor";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    HorizontalViewExtractorParameter _param;
    Frame* _frame;
    HorizontalView* _view;
};

}