// ------------------------------------------------------------------
// Written by Ming li, Baidu, 2017-04-26
// ------------------------------------------------------------------

#ifndef CAFFE_DECONV_ANALYSIS_LAYER_HPP_
#define CAFFE_DECONV_ANALYSIS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_lm_layer.hpp"

namespace caffe {

template <typename Dtype>
class DeconvolutionAnalysisLayer : public BaseConvolutionLmLayer<Dtype> {
    public:
        explicit DeconvolutionAnalysisLayer(const LayerParameter& param)
            : BaseConvolutionLmLayer<Dtype>(param) {}

        virtual inline const char* type() const { return "DeconvolutionAnalysis"; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual inline bool reverse_dimensions() { return true; }
        virtual void compute_output_shape();
};

}  // namespace caffe

#endif  // CAFFE_DECONV_ANALYSIS_LAYER_HPP_
