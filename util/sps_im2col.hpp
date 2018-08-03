// ------------------------------------------------------------------
// Written by Ming li, Baidu, 2017-04-26
// ------------------------------------------------------------------

#ifndef _CAFFE_UTIL_SPS_IM2COL_HPP_
#define _CAFFE_UTIL_SPS_IM2COL_HPP_

namespace caffe {
template <typename Dtype>
void sps_im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_spatial_dim,
    const int kernel_extent_h, const int kernel_extent_w,
    const int* kernel_h_offset, const int* kernel_w_offset,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void sps_col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_spatial_dim,
    const int kernel_extent_h, const int kernel_extent_w,
    const int* kernel_h_offset, const int* kernel_w_offset,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);

template <typename Dtype>
void sps_im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_spatial_dim,
    const int kernel_extent_h, const int kernel_extent_w,
    const int* kernel_h_offset, const int* kernel_w_offset,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void sps_col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_spatial_dim,
    const int kernel_extent_h, const int kernel_extent_w,
    const int* kernel_h_offset, const int* kernel_w_offset,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_SPS_IM2COL_HPP_
