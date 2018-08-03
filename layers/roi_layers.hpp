#ifndef CAFFE_ROI_DATA_LAYERS_HPP_
#define CAFFE_ROI_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"

#include "caffe/util/RectMap.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/benchmark.hpp"

//#include "caffe/layers/pyramid_data_layer.hpp"

#include "caffe/util/benchmark.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include "opencv2/opencv.hpp"

using namespace caffe_fcn_data_layer;
using namespace std;

namespace caffe
{
/**
 *		  Convert detection feature map or ROIs to bboxes(ROIs) and bbox(ROI)_info, which are stored in top[0]
 * 		  and top[1] respectively.
 * 		  case 1: bottom[0] = feature map
 * 		  case 2: bottom[0] = ROIs  bottom[1] = ROI_info
 * 		  Each bbox is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  Each bbox_info is represented as (class_id,  w, h , score);
 */
template <typename Dtype>
class ROIOutputLayer : public Layer <Dtype>
{
public:
	explicit ROIOutputLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
	virtual ~ROIOutputLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){};

	virtual inline const char* type() const { return "ROIOutput"; }
	virtual inline int  MinBottomBlobs() const { return 1; }
	
	// 0410Rui //
	//virtual inline int  MaxBottomBlobs() const { return 2; }
	//virtual inline int  ExactNumTopBlobs() const { return 2; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

	vector<BBox<Dtype> >& GetFilteredBBox(int class_id);
	inline vector<vector<BBox<Dtype> > >& GetFilteredBBox(){return output_bboxes_;}
	inline int GetNumClass(){return num_class_;}
	inline vector<string> GetClassNames(){return class_names_;}

private:
	void GetROIsFromFeatureMap(const vector<Blob<Dtype>*>& bottom);
	void GetROIsFromROIs(const vector<Blob<Dtype>*>& bottom);
	bool is_bottom_rois_;
	vector<vector<int> > ROI_num_id_list_;

protected:
	vector<vector<BBox<Dtype> > > all_candidate_bboxes_;
	vector<bool>                  is_candidate_bbox_selected_;
	vector<vector<BBox<Dtype> > > output_bboxes_;

	int bbox_data_size, bbox_info_size;

	Dtype threshold_;
	bool nms_need_nms_;
	Dtype nms_overlap_ratio_;
	int nms_top_n_;
	bool nms_add_score_;

	int channel_per_scale_;
	int num_class_;
	vector<string> class_names_;

	bool show_time_;
	Dtype time_get_bbox_, time_total_, time_nms_, time_bbox_to_blob_;
	Timer timer_get_bbox_;
	Timer timer_total_;
	Timer timer_nms_;
	Timer timer_bbox_to_blob_;
};


/*
 * 		  Generate ROI labels and target_bbox_diff from ground truth BBox and given ROIs and ROI_info.
 *		  Bottom: blob[0]-> ROIs   blob[1] ->ROI_info  blob[2] -> GT_BOX
 *		  The output has 4 blobs, top[0] is the filtered ROI, and top[1] is the ROI_info.
 *		  top[2] and top[3] output the ROI labels and target_bbox_diff.
 *		  ROI label is a non-negative integer, with 0 indicating background.
 *		  You should be careful that class_id in ROI_info does not contain background class. So the correspondence
 *		  betreen class_id and ROI_label is : ROI_label = class_id +1
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
class ROIDataLayer : public Layer<Dtype>
{
public:
	explicit ROIDataLayer(const LayerParameter& param);

	virtual ~ROIDataLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROIData"; }
	virtual inline int  ExactNumBottomBlobs() const { return 3; }
	virtual inline int  ExactNumTopBlobs() const { return 4; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

protected:
	void FilterInputROIsWithIOU(const vector<Blob<Dtype>*>& bottom);
	bool BalanceROIs();
	void ShuffleIds(vector<int>& ids);

	Dtype pos_iou_ratio_, neg_iou_ratio_;
	int num_class_;
	vector<Dtype> valid_ROIs_;
	vector<Dtype> valid_ROI_info_;
	vector<Dtype> valid_ROIs_label_;
	vector<Dtype> valid_ROIs_bbox_diff_;
	vector<Dtype> valid_IOUs_;

	vector<Dtype> out_ROIs_;
	vector<Dtype> out_ROI_info_;
	vector<Dtype> out_ROIs_label_;
	vector<Dtype> out_ROIs_bbox_diff_;
	vector<int> pos_ids_;
	vector<int> normal_neg_ids_;
	vector<int> hard_neg_ids_;
	vector<int> selected_ids_;

	int ROI_data_length_;
	int ROI_info_length_;
	int GT_data_length_;

	bool need_balance_;
	Dtype hard_threshold_;
	Dtype neg_ratio_;
	Dtype hard_ratio_; /// hard_neg refers to neg samples whose IOU ratio is larger than hard_threshold_
	shared_ptr<Caffe::RNG> prefetch_rng_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Convert ROI output into HeatMap
 *		  The input has 4 blobs, bottom[0] is the filtered ROI, and bottom[1] is the ROI_info.
 *		  bottom[2] and bottom[3] denote the predicted labels and  bbox_diff.
 *		  Predicted label is a class_num +1 array, with the first channel indicating background probability.
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
class ROI2HeatMapLayer : public Layer <Dtype>{
public:
	explicit ROI2HeatMapLayer(const LayerParameter& param)
		  :  Layer<Dtype>(param) {}
	virtual ~ROI2HeatMapLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROI2HeatMap"; }
	virtual inline int  ExactNumBottomBlobs() const { return 4; }
	virtual inline int  ExactNumTopBlobs() const { return 1; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

protected:

	int num_class_, map_h_, map_w_, map_num_;
	int ROI_data_length_, ROI_info_length_;
	ROI2HeatMapParam_LabelType label_type_;

};


/**
 * @brief Refine ROI output by ROI_diff and ROI_predicted
 *		  The input has at least three blobs, bottom[0] is the filtered ROI, and bottom[1] is the ROI_info.
 *		  bottom[2] and bottom[3] denote the bbox_diff and predicted labels.
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
class ROIRefineLayer : public Layer <Dtype>{
public:
	explicit ROIRefineLayer(const LayerParameter& param)
		  :  Layer<Dtype>(param) {}
	virtual ~ROIRefineLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROIRefine"; }
	virtual inline int  MinBottomBlobs() const { return 3; }
	virtual inline int  MaxBottomBlobs() const { return 4; }
	virtual inline int  ExactNumTopBlobs() const { return 2; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

protected:
	int ROI_data_length_, ROI_info_length_, num_class_;
	ROIRefineParam_LabelType label_type_;
	bool  has_ROI_score_;

};


/**
 * @brief Show ROI
 *		  If the input is ROIs, the input has at least two blobs: blob[0] is the input images,  blob[1] contains the ROIs,
 *		  and blob[2] is the ROI_info. blob[3] is optional for ground truth bboxes or ROI_label.
 *		  If the input is heatmap, the input has two blobs: blob[0] is the input images,  blob[1] for heatmap
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
class ROIShowLayer : public Layer <Dtype>{
public:
	explicit ROIShowLayer(const LayerParameter& param)
		  :  Layer<Dtype>(param) {}
	virtual ~ROIShowLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROIShow"; }
	virtual inline int  MinBottomBlobs() const { return 2; }
	virtual inline int  ExactNumTopBlobs() const { return 0; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
protected:

	void Show_ROIs(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);
	void Show_HeatMap(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);

	int heat_map_a_, heat_map_b_;
	int input_w_, input_h_,input_num_;
	bool has_the_fourth_blob_;
	bool the_fourth_blob_ROI_label_;
	Dtype mean_bgr_[3];
	string show_output_path_;

	int img_count;
	bool is_input_heatmap_;
	Dtype heatmap_threshold_;
};
    
    // modified by mingli, change ROIOutputLayer to ROIOoutputSSDLayer
    template <typename Dtype>
    class ROIOutputSSDLayer : public Layer <Dtype> {
        public:
            explicit ROIOutputSSDLayer(const LayerParameter& param) : Layer<Dtype>(param) {
            }
            virtual ~ROIOutputSSDLayer(){
            };
            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);//check
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top){
            };

            virtual inline const char* type() const { 
                return "ROIOutputSSD"; 
            }
            virtual inline int  MinBottomBlobs() const { 
                return 1; 
            }
            virtual inline int  MaxBottomBlobs() const { 
                return 2; 
            }
            virtual inline int  ExactNumTopBlobs() const { 
                return 2; 
            }

            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top){
            };
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top){
            };

            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            }
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            }

            inline vector<BBox<Dtype> >& GetFilteredBBox(int class_id) {
                return output_bboxes_[class_id];
            }
            inline vector<vector< BBox<Dtype> > >& GetFilteredBBox() {
                return output_bboxes_;
            }
            inline int GetNumClass() {
                return num_class_;
            }
            inline vector<string> GetClassNames() {
                return class_names_;
            }

        protected:
            vector<vector<BBox<Dtype> > > all_candidate_bboxes_;
            vector<bool> is_candidate_bbox_selected_;
            vector< vector< BBox<Dtype> > > output_bboxes_;
            //int bbox_data_size, bbox_info_size;
            vector<Dtype> threshold_;
            bool nms_need_nms_;
            vector<Dtype> nms_overlap_ratio_;
            vector<int> nms_top_n_;
            // added by liming
            vector<int> nms_max_candidate_n_;
            vector<bool> nms_use_soft_nms_;
            Dtype threshold_objectness_;
            bool nms_among_classes_;
            vector<bool> nms_voting_;
            vector<Dtype> nms_vote_iou_;
            bool nms_add_score_;
            bool refine_out_of_map_bbox_;
            int channel_per_scale_;
            int num_class_;
            vector<string> class_names_;
            vector<int> class_indexes_;
            vector<Dtype> heat_map_a_vec_;
            vector<Dtype> heat_map_b_vec_;
            vector<Dtype> anchor_width_vec_;
            vector<Dtype> anchor_height_vec_;
            vector<Dtype> anchor_x1_vec_;
            vector<Dtype> anchor_y1_vec_;
            vector<Dtype> anchor_x2_vec_;
            vector<Dtype> anchor_y2_vec_;
            vector<Dtype> proposal_min_area_vec_;
            vector<Dtype> proposal_max_area_vec_;
            bool bg_as_one_of_softmax_;
            bool use_target_type_rcnn_;
            bool do_bbox_norm_;
            vector<Dtype> bbox_means_;
            vector<Dtype> bbox_stds_;
            Dtype im_width_;
            Dtype im_height_;
            bool rpn_proposal_output_score_;
            bool regress_agnostic_;
            bool show_time_;
            Dtype time_get_bbox_, time_total_, time_nms_, time_bbox_to_blob_;
            Timer timer_get_bbox_;
            Timer timer_total_;
            Timer timer_nms_;
            Timer timer_bbox_to_blob_;
            Dtype allow_border_;
            Dtype allow_border_ratio_;
            bool bbox_size_add_one_;
            Dtype read_width_scale_;
            Dtype read_height_scale_;
            unsigned int read_height_offset_;
            bool zero_anchor_center_;
            Dtype min_size_h_;
            Dtype min_size_w_;
            DetectionOutputSSDParameter_MIN_SIZE_MODE min_size_mode_;
            vector<Dtype> reg_means_;
            vector<Dtype> reg_stds_;
            //kpts params
            bool has_kpts_;
            bool kpts_reg_as_classify_;
            int kpts_exist_bottom_idx_;
            int kpts_reg_bottom_idx_;
            int kpts_classify_width_;
            int kpts_classify_height_;
            bool kpts_do_norm_;
            int kpts_reg_norm_idx_st_;
            vector<int> kpts_st_for_each_class_;
            vector<int> kpts_ed_for_each_class_;
            Dtype kpts_classify_pad_ratio_;
            //atrs params
            bool has_atrs_;
            int atrs_reg_bottom_idx_;
            bool atrs_do_norm_;
            int atrs_reg_norm_idx_st_;
            vector<ATRSParameter_NormType> atrs_norm_type_;
            //ftrs params
            bool has_ftrs_;
            int ftrs_bottom_idx_;
            //spmp params
            bool has_spmp_;
            int spmp_bottom_idx_;
            int num_spmp_; 
            vector<bool> spmp_class_aware_;
            vector<int> spmp_label_width_;
            vector<int> spmp_label_height_;
            vector<Dtype> spmp_pad_ratio_;
            vector<int> spmp_dim_st_;
            vector<int> spmp_dim_;
            int spmp_dim_sum_;
            //cam3d params
            bool has_cam3d_;
            int cam3d_bottom_idx_;
    };
    // end mingli

}  // namespace caffe

#endif  // CAFFE_ROI_DATA_LAYERS_HPP_
