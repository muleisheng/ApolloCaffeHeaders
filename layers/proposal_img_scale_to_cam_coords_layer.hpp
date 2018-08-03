// ------------------------------------------------------------------
// Written by MingLi, baidu, 2017-08-22, liming22@baidu.com
// ------------------------------------------------------------------

#ifndef CAFFE_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_LAYERS_HPP_
#define CAFFE_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/pyramid_layers.hpp"

namespace caffe {

template <typename Dtype>
class ProposalImgScaleToCamCoordsLayer : public Layer<Dtype> {
    public:
        explicit ProposalImgScaleToCamCoordsLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ProposalImgScaleToCamCoords"; }

        virtual inline int MinBottomBlobs() const { return 5; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

        shared_ptr<Caffe::RNG> shuffle_rng_;

        int num_class_;

        vector<int> sub_class_num_class_;
        vector<int> sub_class_bottom_idx_;
        vector<int> sub_class_num_class_pre_sum_;
        int total_sub_class_num_;
        ProposalImgScaleToCamCoordsParameter_NormType prj_h_norm_type_;
        bool has_size3d_and_orien3d_;
        ProposalImgScaleToCamCoordsParameter_OrienType orien_type_;
        set<int> cls_ids_zero_size3d_w_;
        set<int> cls_ids_zero_size3d_l_;
        set<int> cls_ids_zero_orien3d_;
        bool cmp_pts_corner_3d_;
        bool cmp_pts_corner_2d_;
        int num_top_channels_;
        int size3d_h_bottom_idx_;
        int size3d_w_bottom_idx_;
        int size3d_l_bottom_idx_;
        int orien3d_sin_bottom_idx_;
        int orien3d_cos_bottom_idx_;
        int cam_info_idx_st_in_im_info_;

        bool need_ctr_2d_norm_;
        vector<Dtype> ctr_2d_means_;
        vector<Dtype> ctr_2d_stds_;
        bool need_prj_h_norm_;
        vector<Dtype> prj_h_means_;
        vector<Dtype> prj_h_stds_;
        bool need_real_h_norm_;
        vector<Dtype> real_h_means_;
        vector<Dtype> real_h_stds_;
        bool need_real_w_norm_;
        vector<Dtype> real_w_means_;
        vector<Dtype> real_w_stds_;
        bool need_real_l_norm_;
        vector<Dtype> real_l_means_;
        vector<Dtype> real_l_stds_;
        bool need_sin_norm_;
        vector<Dtype> sin_means_;
        vector<Dtype> sin_stds_;
        bool need_cos_norm_;
        vector<Dtype> cos_means_;
        vector<Dtype> cos_stds_;

        bool has_scale_offset_info_;
        Dtype im_width_scale_;
        Dtype im_height_scale_;
        Dtype cords_offset_x_;
        Dtype cords_offset_y_;

        bool bbox_size_add_one_;

        PyramidImageDataParam<Dtype> pyramid_image_data_param_;
};

}  // namespace caffe

#endif  // CAFFE_PROPOSAL_IMG_SCALE_TO_CAM_COORDS_LAYERS_HPP_ 
