//
// Created by chenjiajia on 17-3-1.
//

#ifndef CAFFE_CHANNEL_PERMUTE_LAYER_HPP
#define CAFFE_CHANNEL_PERMUTE_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Permute the input blob by changing the memory order of the data.
 *
 */

// The main function which does the permute.
template<typename Dtype>
class ChannelPermuteLayer : public Layer<Dtype> {
public:
    explicit ChannelPermuteLayer(const LayerParameter &param)
        : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top);

    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "ChannelPermute"; }

    virtual inline int ExactNumBottomBlobs() const { return 1; }

    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

    virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down,
                              const vector<Blob<Dtype> *> &bottom);

    virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down,
                              const vector<Blob<Dtype> *> &bottom);


    int num_channels_ = 0;
    //  vector<int> sorts_;
    Blob<int> sorts_;
    Blob<int> orders_;
    int total_ = 0;

};

}  // namespace caffe
#endif //CAFFE_CHANNEL_PERMUTE_LAYER_HPP
