#pragma once

namespace smartseg {

template <typename Dtype>
class BaseExtractorLayer : public caffe::Layer<Dtype> {
public:
    explicit BaseExtractorLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
protected:
    template<class Features>
    void bind_features(Blob<Dtype>* fea_blob, Features* feas) {
        Dtype* feature_data = fea_blob->mutable_cpu_data();
        CHECK(fea_blob->num() == 1);
        int num_features = fea_blob->channels();
        CHECK(sizeof(Features) == sizeof(Dtype*) * num_features);
        Dtype** ptrs = (Dtype**)(void*)feas;
        for (int i = 0; i < num_features; i++) {
            ptrs[i] = feature_data + fea_blob->offset(0, i);
        }
    }
    float fast_log(int n) const {
        return (n < _log_value_num) ? _log_values[n] : std::log((float)n);
    }
private:
    enum { _log_value_num = 64 };
    float _log_values[_log_value_num];
};

}