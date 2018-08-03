#ifndef CAFFE_CNNSEG_UTIL_HPP
#define CAFFE_CNNSEG_UTIL_HPP

#include <opencv2/opencv.hpp>
#include "caffe/blob.hpp"

namespace caffe {
namespace cnnseg {

bool blob_to_mat(const Blob<double>* blob, int offset, int num_pts, cv::Mat &pc_mat);
bool blob_to_mat(const Blob<float>* blob, int offset, int num_pts, cv::Mat &pc_mat);

inline int pc2pixel(float in_pc, float in_range, float out_size) {
    float inv_res = 0.5 * out_size / in_range;
    return (int)std::floor((in_range - in_pc) * inv_res);
}

inline float pixel2pc(int in_pixel, float in_size, float out_range) {
    float res = 2.0 * out_range / in_size;
    return out_range - ((float)in_pixel + 0.5f) * res;
}

bool cyclinder_pt2img(const cv::Point3f &pt3d, cv::Point2i &pt2d, cv::Size img_size,
                      cv::Point2d center_pt, cv::Vec2d angle_res);

void cyclinder_pc_project(const cv::Mat &pc_mat, cv::Mat &out_data,
                          cv::Size img_size, cv::Point2f center_pt,
                          cv::Vec2f angle_res);

cv::Mat norm_image(cv::Mat &img);

template<typename Dtype>
void transform_data(const cv::Mat &data, Dtype *out_data);

}
}

#endif  // CAFFE_CNNSEG_UTIL_HPP
