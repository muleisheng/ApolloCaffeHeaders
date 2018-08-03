// ------------------------------------------------------------------
// Written by Ming li, Baidu, 2017-04-26
// ------------------------------------------------------------------

#ifndef _CAFFE_UTIL_DFMB_HPP_
#define _CAFFE_UTIL_DFMB_HPP_

namespace caffe {

// cpu
template <typename Dtype>
void dfmb_offset_to_bts_info_cpu(const int conv_out_height, 
    const int conv_out_width, const int kernel_spatial_dim, 
    const Dtype* dfmb_offset, int* idx_bts, Dtype* wt_bts, 
    Dtype* wt_bts_fcty, Dtype* wt_bts_fctx);

template <typename Dtype>
void dfmb_col_forward_cpu(const int conv_in_channels, 
    const int conv_out_height, const int conv_out_width, 
    const int kernel_spatial_dim, const Dtype* col_buff, 
    const int* idx_bts, const Dtype* wt_bts, Dtype* col_buff_bts);

void dfmb_reverse_bts_idx_cpu(const int conv_out_height, 
    const int conv_out_width, const int kernel_spatial_dim, 
    const int* idx_bts, int* _idx_bts_ed, int* _idx_bts);

template <typename Dtype>
void dfmb_col_backward_cpu(const int conv_in_channels,
    const int conv_out_height, const int conv_out_width, 
    const int kernel_spatial_dim, const Dtype* wt_bts, 
    const int* _idx_bts_ed, const int* _idx_bts, 
    const Dtype* col_top_diff, Dtype* col_bottom_diff);

template <typename Dtype>
void dfmb_wbts_backward_cpu(const int conv_in_channels, 
    const int conv_out_height, const int conv_out_width, 
    const int kernel_spatial_dim, const Dtype* col_buff, 
    const int* idx_bts, const Dtype* col_top_diff, 
    Dtype* wt_diff);

template <typename Dtype>
void dfmb_merge_diff_cpu(const int conv_out_height,
    const int conv_out_width, const int kernel_spatial_dim,
    const Dtype* wt_diff_bts_fcty, const Dtype* wt_diff_bts_fctx,
    Dtype* wt_diff);

// gpu
template <typename Dtype>
void dfmb_offset_to_bts_info_gpu(const int conv_out_height, 
    const int conv_out_width, const int kernel_spatial_dim, 
    const Dtype* dfmb_offset, int* idx_bts, Dtype* wt_bts, 
    Dtype* wt_bts_fcty, Dtype* wt_bts_fctx);

template <typename Dtype>
void dfmb_col_forward_gpu(const int conv_in_channels, 
    const int conv_out_height, const int conv_out_width, 
    const int kernel_spatial_dim, const Dtype* col_buff, 
    const int* idx_bts, const Dtype* wt_bts, Dtype* col_buff_bts);

void dfmb_reverse_bts_idx_gpu(const int conv_out_height, 
    const int conv_out_width, const int kernel_spatial_dim, 
    const int* idx_bts, int* _idx_bts_ed, int* _idx_bts);

template <typename Dtype>
void dfmb_col_backward_gpu(const int conv_in_channels,
    const int conv_out_height, const int conv_out_width, 
    const int kernel_spatial_dim, const Dtype* wt_bts, 
    const int* _idx_bts_ed, const int* _idx_bts, 
    const Dtype* col_top_diff, Dtype* col_bottom_diff);

template <typename Dtype>
void dfmb_wbts_backward_gpu(const int conv_in_channels, 
    const int conv_out_height, const int conv_out_width, 
    const int kernel_spatial_dim, const Dtype* col_buff, 
    const int* idx_bts, const Dtype* col_top_diff, 
    Dtype* wt_diff);

template <typename Dtype>
void dfmb_merge_diff_gpu(const int conv_out_height,
    const int conv_out_width, const int kernel_spatial_dim,
    const Dtype* wt_diff_bts_fcty, const Dtype* wt_diff_bts_fctx,
    Dtype* wt_diff);

}  // namespace caffe

#endif  // CAFFE_UTIL_DFMB_HPP_
