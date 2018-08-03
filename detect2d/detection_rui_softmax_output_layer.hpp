#ifndef INCLUDE_CAFFE_LAYERS_DETECTION_RUI_SOFTMAX_OUTPUT_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_DETECTION_RUI_SOFTMAX_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/detect2d/BlockPack.hpp"
#include "caffe/layers/roi_layers.hpp"    //ROIOutputLayer
//#include "caffe/layers/roi_data_layers.hpp"    //ROIOutputLayer
namespace caffe {

template<typename Dtype>
class PyramidRuiSoftmaxImageDataParam {
public:
    int ReadFromSerialized(Blob<Dtype>& blob, int start = 0);

    int img_w_, img_h_;
    int heat_map_a_, heat_map_b_;
    BlockPack<Dtype> block_packer_;
};

template<typename Dtype>
class DetectionRuiSoftmaxOutputLayer: public ROIOutputLayer<Dtype> {
public:
    explicit DetectionRuiSoftmaxOutputLayer(const LayerParameter& param) :
            ROIOutputLayer<Dtype>(param) {
    }
    virtual ~DetectionRuiSoftmaxOutputLayer() {
    }
    ;

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const {
        return "DetectionSoftmaxOutput";
    }

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
    }
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
    }

    inline vector<BBox<Dtype> >& GetFilteredGroupBBox(int class_id) {
        return three_class_output_bboxes_[class_id];
    }
    inline bool ifgroupvehicle() {
        return group_veh_;
    }
    inline int groupnumclass() {
        return group_num_class_;
    }

protected:
    PyramidRuiSoftmaxImageDataParam<Dtype> pyramid_image_data_param_;
    bool refine_out_of_map_bbox_;
    bool all_pos_;
    int step_;

    vector<Dtype> mean_w_;
    vector<Dtype> mean_h_;
    vector<Dtype> mean_l_;

    vector<int> class_inds_;
    vector<int> dim_inds_;
    vector<int> pars_inds_;
    vector<int> angle_inds_;
    vector<int> pts8_inds_;
    vector<int> box2d_inds_;

    ////////
    bool group_veh_;
    int group_num_class_;

    vector<vector<BBox<Dtype> > > three_class_candidate_bboxes_;
    vector<vector<BBox<Dtype> > > three_class_output_bboxes_;

    vector<Dtype> new_class_thr;
    vector<Dtype> new_class_overlap_ratio;

    int shallow_track_step_;

    ////////
    bool use_dim;
    bool use_angle;
    bool use_pts8;
    bool use_pars;

};

}

#endif /* INCLUDE_CAFFE_LAYERS_DETECTION_OUTPUT_LAYER_HPP_ */
