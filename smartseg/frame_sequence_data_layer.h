#pragma once

namespace smartseg {

template <typename Dtype>
class FrameSequenceDataLayer : public caffe::Layer<Dtype> {
public:
    explicit FrameSequenceDataLayer(const caffe::LayerParameter& param)
        : caffe::Layer<Dtype>(param) {
        _param = param.smartseg_frame_sequence_data_param();
        _num_frames = _param.num_frames();
    }
    ~FrameSequenceDataLayer() {
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
        return _num_frames;
    }
    const char* type() const override {
        return "SmartsegFrameSequenceData";
    }
protected:
    void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
        const std::vector<Blob<Dtype>*>& top) override;
    void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
        const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
private:
    FrameSequenceDataParameter _param;
    int _num_frames;
    std::vector<std::string> _sources;
    Channel<FrameSequence> _channel;
    FrameSequence _frame_seq;
    std::thread _thread;

    void run_thread();
};

}