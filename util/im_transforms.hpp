#ifndef IM_TRANSFORMS_HPP
#define IM_TRANSFORMS_HPP

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#ifdef USE_OPENCV

    cv::Mat colorReduce(const cv::Mat& image, int div = 64);

    void constantNoise(const int n, const vector<uchar>& val, cv::Mat* image);

    cv::Mat ApplyNoise(const cv::Mat& in_img, 
            const caffe_fcn_data_layer::NoiseParameter& param);

    void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
            const float brightness_prob, const float brightness_delta);

    void AdjustBrightness(const cv::Mat& in_img, const float delta,
            cv::Mat* out_img);

    void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
            const float contrast_prob, const float lower, const float upper);

    void AdjustContrast(const cv::Mat& in_img, const float delta,
            cv::Mat* out_img);

    void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
            const float saturation_prob, const float lower, const float upper);

    void AdjustSaturation(const cv::Mat& in_img, const float delta,
            cv::Mat* out_img);

    void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
            const float hue_prob, const float hue_delta);

    void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

    cv::Mat ApplyDistort(const cv::Mat& in_img, 
            const caffe_fcn_data_layer::DistortionParameter& param);
#endif  // USE_OPENCV

}  // namespace caffe

#endif  // IM_TRANSFORMS_HPP
