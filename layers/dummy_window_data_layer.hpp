#ifndef CAFFE_WINDOW_DATA_LAYER_HPP_
#define CAFFE_WINDOW_DATA_LAYER_HPP_

#include <algorithm>
#include <map>

#include <string>
#include <utility>
#include <vector>

#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"




namespace caffe
{

template <typename Dtype>
class dummy_WindowDataLayer : public BasePrefetchingDataLayer<Dtype>
{
 public:
  explicit dummy_WindowDataLayer(const LayerParameter& param) : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~dummy_WindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WindowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:

  virtual void load_batch(Batch<Dtype>* batch);

//  shared_ptr<Caffe::RNG> prefetch_rng_;
//  vector<std::pair<std::string, vector<int> > > image_database_;
//  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
//  vector<vector<float> > fg_windows_;
//  vector<vector<float> > bg_windows_;
//  Blob<Dtype> data_mean_;
//  vector<Dtype> mean_values_;
//  bool has_mean_file_;
//  bool has_mean_values_;
//  bool cache_images_;
//  vector<std::pair<std::string, Datum > > image_database_cache_;
};

}  // namespace caffe

#endif  // CAFFE_WINDOW_DATA_LAYER_HPP_
