// ------------------------------------------------------------------
// Written by Ming li, Baidu, 2017-04-26
// ------------------------------------------------------------------

#ifndef CAFFE_CONV_ANALYSIS_LAYER_HPP_
#define CAFFE_CONV_ANALYSIS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_lm_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvolutionAnalysisLayer : public BaseConvolutionLmLayer<Dtype> {
    public:
        explicit ConvolutionAnalysisLayer(const LayerParameter& param)
            : BaseConvolutionLmLayer<Dtype>(param) {}

        virtual inline const char* type() const { return "ConvolutionAnalysis"; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual inline bool reverse_dimensions() { return false; }
        virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_CONV_ANALYSIS_LAYER_HPP_
