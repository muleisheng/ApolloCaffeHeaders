#pragma once

#ifdef USE_OPENCV

namespace smartseg {

template <typename Dtype>
class BlobVisualizeLayer : public caffe::Layer<Dtype> {
public:
    explicit BlobVisualizeLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
        _param = param.smartseg_blob_visualize_param();
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    int ExactNumBottomBlobs() const override {
        return 2;
    }
    int ExactNumTopBlobs() const override {
        return 0;
    }
    const char* type() const override {
        return "SmartsegBlobVisualize";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    BlobVisualizeParameter _param;
    Frame* _frame;
};

}

#endif