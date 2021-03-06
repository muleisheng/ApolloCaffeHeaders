// ------------------------------------------------------------------
// Written by MingLi, liming22@baidu.com, 2017-05-27
// ------------------------------------------------------------------

#ifndef CAFFE_BASE_CONVOLUTION_LM_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/util/sps_im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLmLayer : public Layer<Dtype> {
    public:
        explicit BaseConvolutionLmLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        // Helper functions that abstract away the column buffer and gemm arguments.
        // The last argument in forward_cpu_gemm is so that we can skip the im2col if
        // we just called weight_cpu_gemm with the same input.
        void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
                Dtype* output, bool skip_im2col = false);
        void forward_cpu_bias(Dtype* output, const Dtype* bias);
        void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
                Dtype* output);
        void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
                weights);
        void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
        void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
                Dtype* output, bool skip_im2col = false);
        void forward_gpu_bias(Dtype* output, const Dtype* bias);
        void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
                Dtype* col_output);
        void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
                weights);
        void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

        // added by mingli
        // gemm functions for deep neural analysis.
        // TODO. support deformable.
        void forward_cpu_gemm(const Dtype* input, const Dtype* mask_data, 
                const Dtype* weights, Dtype* output, Dtype* mask_diff = NULL, 
                const Dtype* input_data = NULL, const Dtype* dfmb_data = NULL, 
                Dtype* dfmb_diff = NULL, bool skip_im2col = false);
        void backward_cpu_gemm(const Dtype* input, const Dtype* mask_data,
                const Dtype* weights, Dtype* output, Dtype* mask_diff = NULL,
                const Dtype* output_data = NULL, const Dtype* dfmb_data = NULL, 
                Dtype* dfmb_diff = NULL);
        void weight_cpu_gemm(const Dtype* input, const Dtype* mask_data,
                const Dtype* output, Dtype* weights, const Dtype* dfmb_data = NULL);
        void weight_deconv_cpu_gemm(const Dtype* input, const Dtype* mask_data,
                const Dtype* output, Dtype* weights, const Dtype* dfmb_data = NULL);
#ifndef CPU_ONLY
        void forward_gpu_gemm(const Dtype* col_input, const Dtype* mask_data,
                const Dtype* weights, Dtype* output, Dtype* mask_diff = NULL,
                const Dtype* input_data = NULL, const Dtype* dfmb_data = NULL, 
                Dtype* dfmb_diff = NULL, bool skip_im2col = false);
        void backward_gpu_gemm(const Dtype* input, const Dtype* mask_data,
                const Dtype* weights, Dtype* col_output, Dtype* mask_diff = NULL,
                const Dtype* output_data = NULL, const Dtype* dfmb_data = NULL,
                Dtype* dfmb_diff = NULL);
        void weight_gpu_gemm(const Dtype* col_input, const Dtype* mask_data,
                const Dtype* output, Dtype* weights, const Dtype* dfmb_data = NULL);
        void weight_deconv_gpu_gemm(const Dtype* col_input, const Dtype* mask_data,
                const Dtype* output, Dtype* weights, const Dtype* dfmb_data = NULL);
#endif
        // end mingli

        /// @brief The spatial dimensions of the input.
        inline int input_shape(int i) {
            return (*bottom_shape_)[channel_axis_ + i];
        }
        // reverse_dimensions should return true iff we are implementing deconv, so
        // that conv helpers know which dimensions are which.
        virtual bool reverse_dimensions() = 0;
        // Compute height_out_ and width_out_ from other parameters.
        virtual void compute_output_shape() = 0;

        /// @brief The spatial dimensions of a filter kernel.
        Blob<int> kernel_shape_;
        /// @brief The spatial dimensions of the stride.
        Blob<int> stride_;
        /// @brief The spatial dimensions of the padding.
        Blob<int> pad_;
        /// @brief The spatial dimensions of the dilation.
        Blob<int> dilation_;
        /// @brief The spatial dimensions of the convolution input.
        Blob<int> conv_input_shape_;
        /// @brief The spatial dimensions of the col_buffer.
        vector<int> col_buffer_shape_;
        /// @brief The spatial dimensions of the output.
        vector<int> output_shape_;
        const vector<int>* bottom_shape_;

        int num_spatial_axes_;
        int bottom_dim_;
        int top_dim_;

        int channel_axis_;
        int num_;
        int channels_;
        int group_;
        int out_spatial_dim_;
        int weight_offset_;
        int num_output_;
        bool bias_term_;
        bool is_1x1_;
        bool force_nd_im2col_;
        // added by mingli 
        bool print_weight_bias_statistics_;
        int mask_offset_;
        int dfmb_offset_;
        int dfmb_bottom_idx_;
        int mask_bottom_idx_;
        Blob<int> kernel_h_offset_;
        Blob<int> kernel_w_offset_;
        bool is_sps_;
        // end mingli

    private:
        // wrap im2col/col2im so we don't have to remember the (long) argument lists
        inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
            if (is_sps_) {
                sps_im2col_cpu(data, conv_in_channels_,
                        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                        kernel_spatial_dim_, kernel_shape_.cpu_data()[0], 
                        kernel_shape_.cpu_data()[1], kernel_h_offset_.cpu_data(),
                        kernel_w_offset_.cpu_data(), pad_.cpu_data()[0], pad_.cpu_data()[1], 
                        stride_.cpu_data()[0], stride_.cpu_data()[1], col_buff);
            } else if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
                im2col_cpu(data, conv_in_channels_,
                        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                        pad_.cpu_data()[0], pad_.cpu_data()[1],
                        stride_.cpu_data()[0], stride_.cpu_data()[1],
                        dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
            } else {
                im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
                        col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                        pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
            }
        }
        inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
            if (is_sps_) {
                sps_col2im_cpu(col_buff, conv_in_channels_,
                        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                        kernel_spatial_dim_, kernel_shape_.cpu_data()[0], 
                        kernel_shape_.cpu_data()[1], kernel_h_offset_.cpu_data(),
                        kernel_w_offset_.cpu_data(), pad_.cpu_data()[0], pad_.cpu_data()[1],
                        stride_.cpu_data()[0], stride_.cpu_data()[1], data);
            } else if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
                col2im_cpu(col_buff, conv_in_channels_,
                        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                        pad_.cpu_data()[0], pad_.cpu_data()[1],
                        stride_.cpu_data()[0], stride_.cpu_data()[1],
                        dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
            } else {
                col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
                        col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                        pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
            }
        }
#ifndef CPU_ONLY
        inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
            if (is_sps_) {
                sps_im2col_gpu(data, conv_in_channels_,
                        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                        kernel_spatial_dim_, kernel_shape_.cpu_data()[0], 
                        kernel_shape_.cpu_data()[1], kernel_h_offset_.gpu_data(),
                        kernel_w_offset_.gpu_data(), pad_.cpu_data()[0], pad_.cpu_data()[1], 
                        stride_.cpu_data()[0], stride_.cpu_data()[1], col_buff);
            } else if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
                im2col_gpu(data, conv_in_channels_,
                        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                        pad_.cpu_data()[0], pad_.cpu_data()[1],
                        stride_.cpu_data()[0], stride_.cpu_data()[1],
                        dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
            } else {
                im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
                        conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                        kernel_shape_.gpu_data(), pad_.gpu_data(),
                        stride_.gpu_data(), dilation_.gpu_data(), col_buff);
            }
        }
        inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
            if (is_sps_) {
                sps_col2im_gpu(col_buff, conv_in_channels_,
                        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                        kernel_spatial_dim_, kernel_shape_.cpu_data()[0], 
                        kernel_shape_.cpu_data()[1], kernel_h_offset_.gpu_data(),
                        kernel_w_offset_.gpu_data(), pad_.cpu_data()[0], pad_.cpu_data()[1],
                        stride_.cpu_data()[0], stride_.cpu_data()[1], data);
            } else if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
                col2im_gpu(col_buff, conv_in_channels_,
                        conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                        kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                        pad_.cpu_data()[0], pad_.cpu_data()[1],
                        stride_.cpu_data()[0], stride_.cpu_data()[1],
                        dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
            } else {
                col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
                        conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                        kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
                        dilation_.gpu_data(), data);
            }
        }
#endif

        int num_kernels_im2col_;
        int num_kernels_col2im_;
        int conv_out_channels_;
        int conv_in_channels_;
        int conv_out_spatial_dim_;
        int kernel_dim_;
        int col_offset_;
        int output_offset_;
        Blob<Dtype> col_buffer_;
        Blob<Dtype> bias_multiplier_;
        // added by mingli
        int analysis_num_group_;
        int kernel_spatial_dim_;
        int col_buffer_size_;
        int conv_out_height_;
        int conv_out_width_;
        // mask
        Blob<Dtype> col_mask_;
        Blob<Dtype> mask_multiplier_;
        // dfmb
        Blob<Dtype> col_dfmb_;
        Blob<int> idx_bts_;
        Blob<Dtype> wt_bts_;
        Blob<Dtype> wt_bts_fcty_;
        Blob<Dtype> wt_bts_fctx_;
        Blob<int> _idx_bts_;
        Blob<int> _idx_bts_ed_;
        // end mingli
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LM_LAYER_HPP_
