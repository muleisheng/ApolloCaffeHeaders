#pragma once

namespace smartseg {

template <typename Dtype>
class PlanViewExtractorLayer : public BaseExtractorLayer<Dtype> {
public:
    struct Features {
        Dtype* nonempty;
        Dtype* log_count;
        Dtype* direction;
        Dtype* inverse_distance;
        Dtype* top_height;
        Dtype* mean_height;
        Dtype* top_intensity;
        Dtype* mean_intensity;
    };
    struct SubFeatures {
        Dtype* nonempty;
        Dtype* top_height;
    };
    enum { num_features = sizeof(Features) / sizeof(Dtype*) };
    enum { num_sub_features = sizeof(SubFeatures) / sizeof(Dtype*) };

    explicit PlanViewExtractorLayer(const caffe::LayerParameter& param)
        : BaseExtractorLayer<Dtype>(param) {
        _param = param.smartseg_plan_view_extractor_param();
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    int ExactNumBottomBlobs() const override {
        return 2;
    }
    /* int ExactNumTopBlobs() const override {
        return 1;
    } */
    int MinTopBlobs() const override {
        return 1;
    }
    int MaxTopBlobs() const override {
        return 2;
    }
    const char* type() const override {
        return "SmartsegPlanViewExtractor";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    PlanViewExtractorParameter _param;
    Frame* _frame;
    PlanView* _view;
};

}