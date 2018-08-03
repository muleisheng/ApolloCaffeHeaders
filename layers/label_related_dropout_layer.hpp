#ifndef CAFFE_DECONV_LAYER_HPP_
#define CAFFE_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/common.hpp"

#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/opencv.hpp>

namespace caffe
{

class Point4D
{
public:
	int n,c,y,x;
	Point4D (int n_, int c_, int y_, int x_)
	{
		n = n_;
		c = c_;
		y = y_;
		x = x_;
	}
};

/*
 * @brief  aka hard negative mining layer.It also offers capability to balance
 * 		   negative and positive samples. In current implementation, ignore flags
 * 		   for feature map is also supported.
 *
 * @param bottom		   bottom[0] is the feature map
 * 		   				   you want to perform negative mining.
 * 		   				   bottom[1] is the label.
 * 		   				   The ratio of hard negative sample, random negative sample
 * 		  				   is controlled bu negative_slope_ and hard_ratio_;
 * 		  				   ignore_largest_n will ignore the n  largest mismatched sample .
 * 		  				   bottom[2] is the ignore flags. It should have the same structure as both
 * 		  				   bottom[0]  and bottom[1]..
 *
 * */
template <typename Dtype>
class LabelRelatedDropoutLayer : public Layer <Dtype>
{
public:
	explicit LabelRelatedDropoutLayer(const LayerParameter & param) : Layer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "LabelRelatedDropout"; }
	virtual inline int MinBottomBlobs() const { return 2; }
	virtual inline int MaxBottomBlobs() const { return 3; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

	static bool comparePointScore(const std::pair< Point4D ,Dtype>& c1, const std::pair< Point4D ,Dtype>& c2);

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	void set_mask_for_positive_cpu(const vector<Blob<Dtype>*>& bottom);
	void set_mask_for_positive_gpu(const vector<Blob<Dtype>*>& bottom);

	void set_mask_from_labels_cpu(const vector<Blob<Dtype>*>& bottom);
	void get_all_pos_neg_instances(const vector<Blob<Dtype>*>& bottom, vector<int>& pos_count_in_channel );
	vector<int> get_permutation(int start, int end, bool random);
	void PrintMask();

	Blob<int> mask_vec_; ///< the mask for backpro

	Dtype negative_ratio_;
	Dtype hard_ratio_;
	vector< vector <std::pair< Point4D ,Dtype> > > negative_points_;
	Dtype value_masked_; ///< the value of masked point in feature map is set to value_masked_
	int ignore_largest_n;
	int num_;
	int channels_;
	int height_;
	int width_;
	int margin_;
	bool pic_print_;
	string show_output_path_;
	int min_neg_nums_;
};

}  // namespace caffe

#endif  // CAFFE_DECONV_LAYER_HPP_
