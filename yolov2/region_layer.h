//
// Created by chenjiajia on 17-1-23.
//

#ifndef CAFFE_REGION_LAYER_H
#define CAFFE_REGION_LAYER_H

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/yolov2/bbox_util.hpp"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template<typename Dtype>
class RegionLayer : public LossLayer<Dtype> {
public:
    explicit RegionLayer(const LayerParameter &param)
        : LossLayer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top);

    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "Region"; }

    virtual inline int ExactNumBottomBlobs() const { return 4; }

    virtual inline int ExactNumTopBlobs() const { return 1; }

    void UseTrick(bool flag) {
        is_trick_ = flag;
    }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

    virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

    /// @brief Not implemented
    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down,
                              const vector<Blob<Dtype> *> &bottom) {
        return;
    }

    float
    delta_region_box(const NormalizedBBox &truth, const Dtype *x, int n, int index, int i, int j,
                     int w, int h, Dtype *delta, float scale);

    void
    get_region_box(NormalizedBBox &bbox, const Dtype *loc_data, int i, int j, int w, int h, int n,
                   int index);

    void
    delta_region_class(Dtype *output, Dtype *delta, int index, int cls, int classes, float scale,
                       Dtype *avg_cat);

    vector<AnchorBox> anchor_boxes_;
    vector<float> thresholds_;
    unsigned int object_scale_ = 1;
    unsigned int noobject_scale_ = 1;
    unsigned int class_scale_ = 1;
    unsigned int coord_scale_ = 1;
    bool rescore_ = false;
    unsigned int num_class_ = 1;
    unsigned int num_anchor_ = 5;
    unsigned int num_coord_ = 4;
    unsigned int num_gt_ = 30 * 5;
    unsigned int height_ = 13;
    unsigned int width_ = 13;
    unsigned int iter_cnt_ = 0;
    bool is_trick_ = true;
    bool bias_match_ = false;
};

}  // namespace caffe

#endif //CAFFE_REGION_LAYER_H
