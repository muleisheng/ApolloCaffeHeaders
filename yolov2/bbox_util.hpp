#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_BBOX_UTIL_H_
#define CAFFE_UTIL_BBOX_UTIL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {
// Function used to sort NormalizedBBox, stored in STL container (e.g. vector),
// in descend order based on the score value.
bool SortBBoxDescend(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template<typename T>
bool SortScorePairAscend(const pair<float, T> &pair1,
                         const pair<float, T> &pair2);

// Function sued to sort pair<float, T>, stored in STL container (e.g. vector)
// in descend order based on the score (first) value.
template<typename T>
bool SortScorePairDescend(const pair<float, T> &pair1,
                          const pair<float, T> &pair2);

// Compute bbox size.
float BBoxSize(const NormalizedBBox &bbox, const bool normalized = true);

template<typename Dtype>
Dtype BBoxSize(const Dtype *bbox, const bool normalized = true);

template<typename Dtype>
void GetGroundTruth(const Dtype *gt_data, const int num_gt,
                    map<int, vector<NormalizedBBox> > &all_gt_bboxes);

// Compute the jaccard (intersection over union IoU) overlap between two bboxes.
float JaccardOverlap(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2,
                     const bool normalized = true);

template<typename Dtype>
Dtype JaccardOverlap(const Dtype *bbox1, const Dtype *bbox2);

// Compute the intersection between two bboxes.
void IntersectBBox(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2,
                   NormalizedBBox *intersect_bbox);

// Get top_k scores with corresponding indices.
//    scores: a set of scores.
//    indices: a set of corresponding indices.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
void GetTopKScoreIndex(const vector<float> &scores, const vector<int> &indices,
                       const int top_k, vector<pair<float, int> > *score_index_vec);

// Get max scores with corresponding indices.
//    scores: a set of scores.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
void GetMaxScoreIndex(const vector<float> &scores, const float threshold,
                      const int top_k, vector<pair<float, int> > *score_index_vec);

// Get max scores with corresponding indices.
//    scores: an array of scores.
//    num: number of total scores in the array.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
template<typename Dtype>
void GetMaxScoreIndex(const Dtype *scores, const int num, const float threshold,
                      const int top_k, vector<pair<Dtype, int> > *score_index_vec);

// Get max scores with corresponding indices.
//    scores: a set of scores.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
void GetMaxScoreIndex(const vector<float> &scores, const float threshold,
                      const int top_k, vector<pair<float, int> > *score_index_vec);

// Do non maximum suppression given bboxes and scores.
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    threshold: the threshold used in non maximum suppression.
//    top_k: if not -1, keep at most top_k picked indices.
//    reuse_overlaps: if true, use and update overlaps; otherwise, always
//      compute overlap.
//    overlaps: a temp place to optionally store the overlaps between pairs of
//      bboxes if reuse_overlaps is true.
//    indices: the kept indices of bboxes after nms.
void ApplyNMS(const vector<NormalizedBBox> &bboxes, const vector<float> &scores,
              const float threshold, const int top_k, const bool reuse_overlaps,
              map<int, map<int, float> > *overlaps, vector<int> *indices);

void ApplyNMS(const vector<NormalizedBBox> &bboxes, const vector<float> &scores,
              const float threshold, const int top_k, vector<int> *indices);

void ApplyNMS(const bool *overlapped, const int num, vector<int> *indices);

// Do non maximum suppression given bboxes and scores.
// Inspired by Piotr Dollar's NMS implementation in EdgeBox.
// https://goo.gl/jV3JYS
//    bboxes: a set of bounding boxes.
//    scores: a set of corresponding confidences.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    eta: adaptation rate for nms threshold (see Piotr's paper).
//    top_k: if not -1, keep at most top_k picked indices.
//    indices: the kept indices of bboxes after nms.
void ApplyNMSFast(const vector<NormalizedBBox> &bboxes,
                  const vector<float> &scores, const float score_threshold,
                  const float nms_threshold, const float eta, const int top_k,
                  vector<int> *indices);

// Do non maximum suppression based on raw bboxes and scores data.
// Inspired by Piotr Dollar's NMS implementation in EdgeBox.
// https://goo.gl/jV3JYS
//    bboxes: an array of bounding boxes.
//    scores: an array of corresponding confidences.
//    num: number of total boxes/confidences in the array.
//    score_threshold: a threshold used to filter detection results.
//    nms_threshold: a threshold used in non maximum suppression.
//    eta: adaptation rate for nms threshold (see Piotr's paper).
//    top_k: if not -1, keep at most top_k picked indices.
//    indices: the kept indices of bboxes after nms.
template<typename Dtype>
void ApplyNMSFast(const Dtype *bboxes, const Dtype *scores, const int num,
                  const float score_threshold, const float nms_threshold,
                  const float eta, const int top_k, vector<int> *indices);

#ifndef CPU_ONLY  // GPU

template<typename Dtype>
__host__ __device__ Dtype BBoxSizeGPU(const Dtype *bbox,
                                      const bool normalized = true);

template<typename Dtype>
__host__ __device__ Dtype JaccardOverlapGPU(const Dtype *bbox1,
                                            const Dtype *bbox2);

template<typename Dtype>
void ComputeOverlappedGPU(const int nthreads,
                          const Dtype *bbox_data, const int num_bboxes, const int num_classes,
                          const Dtype overlap_threshold, bool *overlapped_data);

template<typename Dtype>
void ComputeOverlappedByIdxGPU(const int nthreads,
                               const Dtype *bbox_data, const Dtype overlap_threshold,
                               const int *idx, const int num_idx, bool *overlapped_data);

template<typename Dtype>
void ApplyNMSGPU(const Dtype *bbox_data, const Dtype *conf_data,
                 const int num_bboxes, const float confidence_threshold,
                 const int top_k, const float nms_threshold, vector<int> *indices);

#endif
}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
