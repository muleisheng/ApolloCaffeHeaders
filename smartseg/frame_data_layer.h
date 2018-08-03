#pragma once

namespace smartseg {

template <typename Dtype>
class FrameDataLayer : public caffe::Layer<Dtype> {
public:
    explicit FrameDataLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
        _param = param.smartseg_frame_data_param();
    }
    ~FrameDataLayer() {
        _channel.close();
        if (_thread.joinable()) {
            _thread.join();
        }
    }
    void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Reshape(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    int ExactNumBottomBlobs() const override {
        return 0;
    }
    int ExactNumTopBlobs() const override {
        return 1;
    }
    const char* type() const override {
        return "SmartsegFrameData";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    FrameDataParameter _param;
    std::vector<std::string> _sources;
    Channel<Frame> _channel;
    Frame _frame;
    std::thread _thread;

    void run_thread();
};

}