#pragma once

namespace smartseg {

template <typename Dtype>
class PlanViewPointExtractorLayer : public BaseExtractorLayer<Dtype> {
public:
    struct Features {
        Dtype* x;
        Dtype* y;
        Dtype* z;
        Dtype* intensity;
    };
    enum { num_features = sizeof(Features) / sizeof(Dtype*) };

    explicit PlanViewPointExtractorLayer(const caffe::LayerParameter& param)
        : BaseExtractorLayer<Dtype>(param) {
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
        return "SmartsegPlanViewPointExtractor";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    Frame* _frame;
    PlanView* _view;
    int _max_points;
};

}