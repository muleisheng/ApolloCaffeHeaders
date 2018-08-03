#ifndef REORG_LAYER_HPP
#define REORG_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ReorgLayer : public Layer<Dtype> {
public:
    explicit ReorgLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top);
    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "Reorg"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob < Dtype> *> &bottom,
                             const vector<Blob < Dtype> *> &top);
    virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);
    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype> *> &bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype> *> &bottom);

    void reorg(Dtype *bottom, Dtype *top, int batch, bool forward);
    int stride_ = 1;
    int channels_;
    int height_;
    int width_;
};

}  // namespace caffe

#endif //REORG_LAYER_HPP