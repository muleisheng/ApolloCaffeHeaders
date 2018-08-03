#ifndef CAFFE_FCN_DATA_LAYER_HPP_
#define CAFFE_FCN_DATA_LAYER_HPP_

#include <stdint.h>
#include <algorithm>
#include <map>
#include <sys/param.h>

#include <string>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"

#include "caffe/layers/base_data_layer.hpp"

#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/util_others.hpp"

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

using namespace caffe_fcn_data_layer;
using namespace std;

namespace caffe
{
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
class IImageDataSourceProvider
{
public:
	IImageDataSourceProvider(bool need_shuffle = false);
	virtual ~IImageDataSourceProvider(){};
	virtual const std::pair<std::string, vector<Dtype> > &  GetOneSample() = 0;
	virtual void SetShuffleFlag(bool need_shuffle) ;
	virtual bool ReadSamplesFromFile(const string & filename, const string & folder, int num_anno_points_per_instance, bool no_expand = false,int class_flag_id = -1 ) = 0;
	virtual int	 GetSamplesSize()= 0;
	virtual void ShuffleImages()= 0;
	static vector<Dtype> SwapInstance(vector<Dtype>& coords, int num_anno_points_per_instance, int id1, int id2);

protected:
	shared_ptr<Caffe::RNG> prefetch_rng_;
	bool shuffle_;
};


template<typename Dtype>
class ImageDataSourceProvider : virtual public IImageDataSourceProvider<Dtype>
{
	public:
		ImageDataSourceProvider(bool need_shuffle = false);
		virtual ~ImageDataSourceProvider();
		virtual const std::pair<std::string, vector<Dtype> > &  GetOneSample();
		virtual bool ReadSamplesFromFile(const string & filename, const string & folder, int num_anno_points_per_instance, bool no_expand = false,int class_flag_id = -1);
		void SetLineId(int id);
		virtual int	 GetSamplesSize();
		virtual void ShuffleImages();
		void PushBackSample(const pair< string, vector<Dtype> > & cur_sample);
		static string GetSampleName(const pair< string, vector<Dtype> > & cur_sample);
		static string GetSampleFolder(const pair< string, vector<Dtype> > & cur_sample);

	protected:
		vector<pair<string, vector<Dtype> > > samples_;
		vector<int> shuffle_idxs_;
		int lines_id_;
};


template<typename Dtype>
class ImageDataSourceMultiClassProvider : virtual public IImageDataSourceProvider<Dtype>
{
	public:
		ImageDataSourceMultiClassProvider(bool need_shuffle = false);
		virtual ~ImageDataSourceMultiClassProvider();
		virtual const std::pair<std::string, vector<Dtype> > &  GetOneSample();
		/* ReadSamplesFromFile: This function reads annotations from <code>filename</code>, and store all samples.*/
		virtual bool ReadSamplesFromFile(const string & filename, const string & folder, int num_anno_points_per_instance, bool no_expand = false, int class_flag_id = -1  );
		void PushBackSample(const pair< string, vector<Dtype> > & cur_sample, int num_anno_points_per_instance,int class_flag_id );
		virtual int	 GetSamplesSize();
		virtual void ShuffleImages();
	protected:
		/* class_id start from 0. Background  class is -1.  */
		vector<int> class_ids_;
		vector<ImageDataSourceProvider<Dtype> > image_data_providers_;
		int lines_class_id_;
		vector<int> shuffle_class_idxs_;
};


template<typename Dtype>
class ImageDataHardNegSourceProvider : public ImageDataSourceProvider<Dtype>
{
	public:
		ImageDataHardNegSourceProvider(bool need_shuffle = false);
		virtual ~ImageDataHardNegSourceProvider();
		bool ReadHardSamplesFromFile(const string & filename, const string & neg_img_folder, int input_height, int input_width);
		virtual bool ReadSamplesFromFile(const string & filename, const string & folder, int num_anno_points_per_instance, bool no_expand = false,int class_flag_id = -1 )
		{
			return false;
		}
		void SetUpHardNegParam(const FCNImageDataSourceParameter & fcn_img_data_source_param);
	protected:
		Dtype bootstrap_std_length_;
		FCNImageDataSourceParameter_STDLengthType stdLengthType;
};


enum ImageDataSourceSampleType { SOURCE_TYPE_POSITIVE, SOURCE_TYPE_ALL_NEGATIVE, SOURCE_TYPE_HARD_NEGATIVE };


//A source container that feeds instances to DataLayer.	It contains 3 types of samples(positive, all_negative, hard_negative). This container could update hard negative samples on the fly.
template<typename Dtype>
class ImageDataSourceBootstrapableProvider
{
	public:
		ImageDataSourceBootstrapableProvider();
		~ImageDataSourceBootstrapableProvider();
		virtual void SetUpParameter(const FCNImageDataParameter & fcn_image_data_param);
		/* FetchBatchSamples:  fetch all samples needed in the current batch iterations, and store them into <code>cur_batch_samples_</code> . */
		void FetchBatchSamples();
		std::pair<std::string, vector<Dtype> > &  GetMutableSampleInBatchAt(int id);
		ImageDataSourceSampleType& GetMutableSampleTypeInBatchAt(int id);
		bool ReadHardSamplesForBootstrap(const string& detection_result_file, const string & neg_img_folder, int input_height, int input_width );
		int GetBatchSize(){return batch_size_;}
		void ReadPosAndNegSamplesFromFiles(const FCNImageDataSourceParameter & fcn_img_data_source_param, int num_anno_points_per_instance, int class_flag_id = -1);
		int GetTestIterations();
		inline void SetNegRatio(Dtype neg_ratio){neg_ratio_ = neg_ratio;}
	protected:
		void SetUpParameter(const FCNImageDataSourceParameter & fcn_img_data_source_param);
		/*FetchSamplesTypeInBatch: get the sample types in the current batch iteration. The number of different type of samples is calculated according to <code>neg_ratio_</code> and <code>bootstrap_hard_ratio_</code> .*/
		void FetchSamplesTypeInBatch();
		shared_ptr<IImageDataSourceProvider<Dtype> > pos_samples_ptr;
		ImageDataSourceProvider<Dtype>               all_neg_samples;
		ImageDataHardNegSourceProvider<Dtype>        hard_neg_samples;
		Dtype neg_ratio_;
		Dtype bootstrap_hard_ratio_;
		bool shuffle_;
		int batch_size_;
		int batch_pos_count_;
		int batch_neg_count_;
		int batch_hard_neg_count_;
		vector<ImageDataSourceSampleType>     cur_batch_sample_type_ ;
		vector<pair<string, vector<Dtype> > > cur_batch_samples_;
		bool multi_class_sample_balance_;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum ImageDataAnnoType { ANNO_POSITIVE, ANNO_IGNORE, ANNO_NEGATIVE };

//Provides base for all kinds of input processing and label generating, It reads param form FCNImageDataCommonParameter.
//heat_map_a_ heat_map_b_. The coordinate relationship between the input and output has the following mapping: x_in = x_out * heat_map_a_ + b.
template <typename Dtype>
class IImageDataProcessor
{
	public:
		IImageDataProcessor(){total_channel_need_ = 0;}
		virtual ~IImageDataProcessor(){} ;
		virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);
		inline vector<Dtype> GetScaleBase() {return scale_base_;}

	protected:
		void SetUpParameter(const FCNImageDataCommonParameter& fcn_img_data_common_param);
		virtual int SetUpChannelInfo(const int channel_base_offset = 0){return total_channel_need_;}
		string GetPrintSampleName(const pair< string, vector<Dtype> > & cur_sample, const ImageDataSourceSampleType sample_type);
		void SetScaleSamplingWeight();
		int GetWeighedScaleIdByCDF(Dtype point);
		inline bool CheckValidIndexInRange(cv::Mat& cvmat, int tmp_h, int tmp_w)
		{
			return !(tmp_h < 0 || tmp_w < 0 || tmp_h >= cvmat.rows || tmp_w >= cvmat.cols);
		}
		int input_height_;
		int input_width_;
		int heat_map_a_;
		int heat_map_b_;
		int out_height_;
		int out_width_;

		int total_channel_need_;
		int num_anno_points_per_instance_;

		/*
		 * scale_base_: For multi-scale label output. It decides how many sets of label need to generate.
		 * For example, if scale_base_= [1, 2], then the output labels should have one half for scale == 1, and another half for scale == 2.
		 * Suppose a positive instance with scale == k, for each scale_base[i], if k is within the range of positive bounder,
		 * this instance should have positive labels in the ground truth of scale_base[i], or ignore labels if still within the range of ignore bounder.
		 * Instance whose scale is outside the ignore bounder should be treated as negative instance.
		 */
		vector<Dtype> scale_base_;
		vector<Dtype> scale_sampling_weight_;
		FCNImageDataCommonParameter_ScaleChooseStrategy scale_choose_stragety_;

		Dtype scale_positive_upper_bounder_; ///< the upper bounder of scale for a positive instance near one scale_base
		Dtype scale_positive_lower_bounder_; ///< the lower bounder of scale for a positive instance near one scale_base
		Dtype scale_ignore_upper_bounder_; ///< the upper bounder of scale for a positive instance to be ignored
		Dtype scale_ignore_lower_bounder_; ///< the lower bounder of scale for a positive instance to be ignored

		bool pic_print_; ///< Flag for debug. Print the data as well as the label fed to DataLayer in the form of images.
		bool label_print_;///< Flag for debug. Print the data as well as the label fed to DataLayer in the form of text.
		string show_output_path_;

		int PIC_MARGIN;
};


template <typename Dtype>
class IImageDataBoxNorm : virtual public IImageDataProcessor<Dtype>
{
	public:
		IImageDataBoxNorm();
		~IImageDataBoxNorm();
		virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);

		vector<Dtype> GetScalesOfAllInstances(const vector<Dtype> & coords_of_all_instance, int num_points_per_instance, vector<int>& bbox_point_ids);
		vector<ImageDataAnnoType> GetAnnoTypeForAllScaleBase( Dtype scale);
	private:
		void SetUpParameter(const FCNImageDataDetectionBoxParameter& fcn_img_data_detection_box_param){};
		void SetUpParameter(const FCNImageDataBoxNormParameter& fcn_img_data_box_norm_param){};

		Dtype bbox_height_;
		Dtype bbox_width_;
		FCNImageDataBoxNormParameter_BBoxSizeNormType bbox_size_norm_type_;
};


template <typename Dtype>
class IImageDataKeyPoint : virtual public IImageDataProcessor<Dtype>
{
 public:
  IImageDataKeyPoint(){};
  ~IImageDataKeyPoint(){};
  virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);

 protected:
  void SetUpParameter(const FCNImageDataKeyPointParameter& fcn_img_data_keypoint_param);
  virtual int SetUpChannelInfo( const int channel_base_offset = 0);
  void GenerateKeyPointHeatMap(int item_id, vector<float> & coords,vector<float> & box_scale, const LayerParameter& param);
  void PrintPic(int item_id, const string & output_path, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, const pair< string, vector<Dtype> > & cur_sample,const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label);

  int channel_point_base_offset_;
  int channel_point_label_offset_;
  int channel_loc_diff_offset_from_key_point_;
  vector<int> used_key_point_idxs_;
  int key_points_count_;
  int valid_distance_;
  int min_out_valid_len_;
  int ignore_flag_radius_;
  vector<bool> is_key_point_flagged_;
  bool need_point_loc_diff_;
  int  valid_point_loc_diff_dist_;
};


template <typename Dtype>
class IImageDataIgnoreBox : virtual public IImageDataProcessor<Dtype>
{
 public:
  IImageDataIgnoreBox(){};
  ~IImageDataIgnoreBox(){};
  virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);

 protected:
  void SetUpParameter(const FCNImageDataIgnoreBoxParameter& fcn_img_data_ignore_box_param);
  virtual int SetUpChannelInfo( const int channel_base_offset = 0);
  virtual void GenerateIgnoreBoxMap(int item_id, vector<float> & coords,vector<float> & box_scale, const LayerParameter& param);
  void PrintPic(int item_id, const string & output_path, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, const pair< string, vector<Dtype> > & cur_sample, const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label);

  int channel_ignore_box_base_offset_;
  int ignore_box_flag_id_;
  vector<int> ignore_box_point_id_;
};


template <typename Dtype>
class IImageDataReader : virtual public IImageDataProcessor<Dtype>
{
	public:
		IImageDataReader() ;
		virtual ~IImageDataReader();
		virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);

		bool ReadImgAndTransform(int item_id, Blob<Dtype>& prefetch_data, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, pair<string, vector<Dtype> >& mutable_sample, vector<bool>& is_keypoint_transform_ignored, const ImageDataSourceSampleType sample_type, Phase cur_phase,int scale_base_id = 0);
		void PrintPic(int item_id, const string& output_path, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, const pair<string, vector<Dtype> >& cur_sample, const ImageDataSourceSampleType sample_type, const Dtype base_scale, const Blob<Dtype>& prefetch_data);
		Dtype GetRefinedBaseScale(int item_id);//Return the scale of instance w.r.t the normalized scale after transformation.

	protected:
		void SetUpParameter(const FCNImageDataReaderParameter& fcn_img_data_reader_param);

		void SetResizeScale(int item_id, cv::Mat & cv_img_original_no_scaled, const ImageDataSourceSampleType sample_type, const vector<Dtype>& coords, bool& is_neg, int scale_base_id);
		void SetCropAndPad(int item_id, const cv::Mat& cv_img_original, cv::Mat & cv_img, bool is_neg);
		void RefineCoords(int item_id, vector<Dtype>& coords, vector<bool>& is_keypoint_ignored, const ImageDataSourceSampleType sample_type);
		void Rotate(cv::Mat& cv_img,vector<Dtype>& coords, const vector<bool>& is_keypoint_ignored);
		unsigned int PrefetchRand();
		Dtype PrefetchRandFloat();

		Phase cur_phase_;
		Dtype scale_lower_limit_; //lower bounder for scaling augmentation
		Dtype scale_upper_limit_; //upper bounder for scaling augmentation
		int standard_len_point_1_;//point for scale normalization
		int standard_len_point_2_;//point for scale normalization
		int standard_len_;        //normalize the scale of input image so that the distance of standard_len_point_1_ and standard_len_point_2_ is standard_len_.
		int roi_center_point_;    //point id of the roi center.
		int rand_x_perturb_;
		int rand_y_perturb_;
		vector<pair<int, int> > crop_begs_;
		vector<pair<int, int> > paddings_;
		vector<Dtype> standard_scales_;
		vector<Dtype> random_scales_;
		vector<Dtype> sample_scales_;
		vector<Dtype> center_x_;
		vector<Dtype> center_y_;
		vector<Dtype> lt_x_;
		vector<Dtype> lt_y_;
		Dtype random_rotate_degree_;
		Dtype mean_bgr_[3];
		shared_ptr<Caffe::RNG> prefetch_rng_;
		Dtype coord_jitter_; //Set coords to coods + coord_jitter_*standard_len_*rand(0-1)
		Dtype random_roi_prob_; //The probability of random set the roi in the image
		boost::shared_mutex mutex_;
};


template <typename Dtype>
class IImageDataDetectionBox : virtual public IImageDataProcessor<Dtype>
{
	public:
		IImageDataDetectionBox() ;
		~IImageDataDetectionBox(){} ;
		virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);

		void GenerateDetectionMap(int item_id, const vector<Dtype> & coords_of_all_instance, Blob<Dtype>& prefetch_label, int used_scale_base_id);
		vector<Dtype> GetScalesOfAllInstances(const vector<Dtype> & coords_of_all_instance);

		void PrintPic(int item_id, const string & output_path, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, const pair< string, vector<Dtype> > & cur_sample,const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label);
		void PrintLabel(int item_id,const string & output_path,const pair< string, vector<Dtype> > & cur_sample, const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label);
		void ShowDataAndPredictedLabel(const string & output_path,const string & img_name, const Blob<Dtype>& data, const int sampleID,const Dtype* mean_bgr,const Blob<Dtype>& label,const Blob<Dtype>& predicted, Dtype threshold);

		void GTBBoxesToBlob(Blob<Dtype>& prefetch_bbox);//thread unsafe function
		void PrintBBoxes(const Blob<Dtype>& prefetch_bbox);

		inline int GetDetectionBaseChannelOffset(){return channel_detection_base_channel_offset_;}
		inline int GetDetectionLabelChannelOffset(){return channel_detection_label_channel_offset_;}
		inline int GetDetectionDiffChannelOffset(){return channel_point_diff_from_center_channel_offset_;}

	protected:
		void SetUpParameter(const FCNImageDataDetectionBoxParameter& fcn_img_data_detection_box_param);
		virtual int SetUpChannelInfo(const int channel_base_offset = 0);

		vector<ImageDataAnnoType> GetAnnoTypeForAllScaleBase( Dtype scale);

		void GenerateDetectionMapForOneInstance(int item_id, const vector<Dtype> & coords_of_one_instance, const Dtype scale, const vector<ImageDataAnnoType> anno_type, Blob<Dtype>& prefetch_label, int used_scale_base_id);

		void LabelToVisualizedCVMat(const Blob<Dtype>& label, const int class_id,cv::Mat& out_probs, cv::Mat& ignore_out_probs, int item_id, int scale_base_id, Dtype* color_channel_weight, Dtype threshold, bool need_regression = true,Dtype heatmap_a = 1, Dtype heatmap_b =  0);

		bool IsLabelMapAllZero(const Blob<Dtype>& label, const int class_id,int item_id, int scale_base_id);

		int channel_detection_base_channel_offset_;
		int channel_detection_label_channel_offset_;
		int channel_detection_ignore_label_channel_offset_;
		int channel_detection_diff_channel_offset_;
		int channel_point_diff_from_center_channel_offset_;
		int channel_detection_channel_all_need_;
		int total_class_num_;
		int class_flag_id_;
		bool loc_regress_on_ignore_;
		int min_output_pos_radius_; //minimum radius of positive instance in the ground truth
		int ignore_margin_;
		Dtype bbox_height_;
		Dtype bbox_width_;
		vector<int> bbox_point_id_;                       //left top and right bottom of bbox.
		Dtype bbox_valid_dist_ratio_;                     //the radius of positive region is calculated as bbox_valid_dist_ratio_ * bbox_height_  or bbox_valid_dist_ratio_ * bbox_width_
		bool need_detection_loc_diff_;                    //indicate whether to generate bbox regression
		Dtype bbox_loc_diff_valid_dist_ratio_;            //the radius of bbox regression region is calculated as bbox_loc_diff_valid_dist_ratio_ * bbox_height_ or bbox_valid_dist_ratio_ * bbox_width_
		bool need_point_diff_from_center_;                //indicate whether to regress the location of landmark.
		vector<int> point_id_for_point_diff_from_center_; //point id for location regression.

		FCNImageDataDetectionBoxParameter_BBoxSizeNormType bbox_size_norm_type_;
		vector<vector<Dtype> > gt_bboxes_; //Each bbox is represented as: (class_id, item_id,roi_start_w, roi_start_h, roi_end_w, roi_end_w)
		boost::shared_mutex bbox_mutex_;
		bool bbox_print_;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
class FCNImageDataLayer : public BasePrefetchingDataLayer<Dtype>, public IImageDataIgnoreBox<Dtype>, public IImageDataKeyPoint<Dtype>, public IImageDataDetectionBox<Dtype>, public IImageDataReader<Dtype>
{
public:
	explicit FCNImageDataLayer(const LayerParameter& param) : BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~FCNImageDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "FCNImageData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 2; }

protected:
	virtual void load_batch(Batch<Dtype>* batch);

	virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param){};
	virtual int  SetUpChannelInfo(const int channel_base_offset = 0){return 0;};

	int	GetScaleBaseId();

	/////////BaseImageDataLayer<Dtype>::DataLayerSetUp(bottom, top)/////////
	int   batch_size_;
/*
	bool  single_thread_;
	Blob<Dtype> prefetch_bbox_;
	bool        need_prefetch_bbox_;
*/
	/////////ImageDataSourceBootstrapableProvider/////////
	ImageDataSourceBootstrapableProvider<Dtype> data_provider_;

	/////////FCNImageDataLayer/////////
	bool need_detection_box_;
	bool need_keypoint_;
	bool need_ignore_box_;
	vector<int> chosen_scale_id_;
};


}
#endif
