#ifndef CAFFE_DEFORMABLE_CONV_LAYER_HPP_
#define CAFFE_DEFORMABLE_CONV_LAYER_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_deformable_conv_layer.hpp"

namespace caffe {
template<typename Dtype>
class DeformableConvolutionLayer: public BaseDeformableConvolutionLayer<Dtype> {
public:
    explicit DeformableConvolutionLayer(const LayerParameter& param) :
            BaseDeformableConvolutionLayer<Dtype>(param) {
        this->is_deformable_conv_ = true;
        deformable_group_ = 1;
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const {
        return "DeformableConvolution";
    }
    virtual inline bool EqualNumBottomTopBlobs() const {
        return false;
    }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
    virtual inline bool reverse_dimensions() {
        return false;
    }
    virtual void compute_output_shape();

private:
    void deformable_im2col(const Dtype* data_im, const Dtype* data_offset,
            const vector<int>& im_shape, const vector<int>& col_shape,
            const Blob<int>& kernel_shape, const Blob<int>& pad, const Blob<int>& stride,
            const Blob<int>& dilation, const int deformable_group, Dtype* data_col);

    void deformable_col2im_coord(const Dtype* data_col, const Dtype* data_im,
            const Dtype* data_offset, const vector<int>& im_shape, const vector<int>& col_shape,
            const Blob<int>& kernel_shape, const Blob<int>& pad, const Blob<int>& stride,
            const Blob<int>& dilation, const int& deformable_group, Dtype* grad_offset);

    void deformable_col2im(const Dtype* data_col, const Dtype* data_offset,
            const vector<int>& im_shape, const vector<int>& col_shape,
            const Blob<int>& kernel_shape, const Blob<int>& pad, const Blob<int>& stride,
            const Blob<int>& dilation, const int& deformable_group, Dtype* grad_im);

#ifndef CPU_ONLY
    void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights, Dtype* output);
    void backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input);
    void weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights, bool is_first =
            false);
    void backward_gpu_bias(Dtype* bias, const Dtype* input, bool is_first = false);
#endif

    int input_offset_dim_;
    const vector<int>* offset_shape_;
    vector<int> deform_col_buffer_shape_;
    Blob<Dtype> deform_col_buffer_;
    int deformable_group_;

};
}
#endif
