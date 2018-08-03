/*************************************************************************
	> File Name: util_traffic_lights_recognition.hpp
	> Author: chenshijia
	> Mail: chenshijia@baidu.com 
	> Created Time: Mon 01 Feb 2016 03:52:08 PM CST
 ************************************************************************/

#ifndef CAFFE_UTIL_UTIL_TRAFFIC_LIGHTS_RECOGNITION_H_
#define CAFFE_UTIL_UTIL_TRAFFIC_LIGHTS_RECOGNITION_H_

#include <string>
#include <map>
#include <utility>
#include <vector>

#define DEBUG_UTIL_TRAFFIC_LIGHTS_RECOGNITION 0

namespace caffe {

typedef struct diff_mat_struct {
    int first_id;
    int second_id;
    cv::Mat diff;
}SDiffMat;

typedef struct scan_line_struct {
    int orientation;  // 0 horizontal, 1 vertical
    int start;
    int end;
    int type; // 0 black or 1 white
    int length;
}SScanLine;

typedef vector<vector<cv::Point> > SContour;
enum light_type { UNKNOWN, CIRCLE, LEFT_ARROW, RIGHT_ARROW, FORWARD_ARROW };

bool mean_value_comp(const std::pair<int, cv::Scalar> &value_pair1, const std::pair<int, cv::Scalar> &value_pair2) {
    return (value_pair1.second[0] > value_pair2.second[0]);
}

inline int get_greater_than_diff(const cv::Mat &mat1, const cv::Mat &mat2, cv::Mat &diff_mat) {
    //assert() check size
    //check single channel

    int width = mat1.cols;
    int height = mat1.rows;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int diff_value = int(mat1.at<uchar>(i, j)) - int(mat2.at<uchar>(i, j));
            diff_mat.at<uchar>(i, j) = diff_value > 0 ? (uchar(diff_value)) : 0;
        }
    }

    return 0;
}

int generate_scan_base_locat(const cv::Mat &binary_mat, const int hor_scan_line_num, const int ver_scan_line_num, vector<std::pair<int, int> > &base_locate_vec);

int get_scan_line_fragments(const cv::Mat &binary_mat, std::pair<int, int> &scan_line_local, vector<SScanLine> &scan_line_fragment_vec);


int reconstruct_fragment_set(const int img_width, const int img_height, vector<SScanLine> &scan_line_vec, const float length_thres);

int draw_scan_lines(const cv::Mat &src_mat, const vector<std::pair<int, int> > &base_locate_vec, const vector<vector<SScanLine> > &scan_line_fragment_vec, const string save_scan_line_prefix);

bool fragment_length_comp(const std::pair<int, float> &frag1, const std::pair<int, float> &frag2) {
    return (frag1.second > frag2.second);
}

#if DEBUG_UTIL_TRAFFIC_LIGHTS_RECOGNITION
int coarse_filter_candidate_boxes_by_rules(vector<std::pair<cv::Rect, int> > &box_candidate_vec, const SContour &contours, const int img_width, const int img_height, const string image_name);
#else
int coarse_filter_candidate_boxes_by_rules(vector<std::pair<cv::Rect, int> > &box_candidate_vec, const SContour &contours, const int img_width, const int img_height);
#endif

inline int draw_boxes(const cv::Mat src_mat, const vector<std::pair<cv::Rect, int> > &box_candidate_vec, const string box_draw_mat_path);

inline int draw_contours(const cv::Mat &src_mat, const SContour &contours, const string contour_draw_mat_path_prefix);

#if DEBUG_UTIL_TRAFFIC_LIGHTS_RECOGNITION
int get_candidate_boxes(const cv::Mat binary_mat, vector<std::pair<cv::Rect, int> > &box_candidate_vec, string image_name);
#else
int get_candidate_boxes(const cv::Mat binary_mat, vector<std::pair<cv::Rect, int> > &box_candidate_vec);
#endif

bool block_box_comp(const std::pair<cv::Rect, int> &box1, const std::pair<cv::Rect, int> &box2) {
    int area1 = box1.first.width * box1.first.height;
    int area2 = box2.first.width * box2.first.height;
    return (area1 > area2);
}

inline int need_nms(const cv::Rect &rect1, const cv::Rect &rect2) {
    int max_left = std::max(rect1.x, rect2.x);
    int max_top = std::max(rect1.y, rect2.y);
    int min_right = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    int min_bottom = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    if (max_left > min_right || max_top > min_bottom) { return 0; } // not overlapping

    int inner_area = (min_right - max_left) * (min_bottom - max_top);
    int area1 = rect1.width * rect1.height;
    int area2 = rect2.width * rect2.height;
    int union_area = area1 + area2 - inner_area;
    float single_occupy_thres = 0.8f;
    float iou_thres = 0.7f;
    if (inner_area > std::min(area1, area2) * single_occupy_thres) {
        // one rect is mainly in the other
        return 1; 
    } else if (inner_area > union_area * iou_thres) {
        return 1;
    }

    return 0;
}


int generate_light_candidates(const vector<cv::Rect> &block_roi_vec,
        vector<vector<std::pair<cv::Rect, int> > > &block_mat_box_candidate_vec,
        vector<std::pair<cv::Rect, int> > &box_candidate_vec, int nms);

int draw_recognition_result(const cv::Mat &src_mat, const vector<std::pair<cv::Rect, int> > &box_candidate_vec, const string save_img_name);

#if DEBUG_UTIL_TRAFFIC_LIGHTS_RECOGNITION
int traffic_lights_recognition_by_blocks(const cv::Mat &src_mat, const int block_x, const int block_y, 
        const float diff_thres, vector<std::pair<cv::Rect, int> > &box_candidate_vec, string img_name);
#else
int traffic_lights_recognition_by_blocks(const cv::Mat &src_mat, const int block_x, const int block_y, 
        const float diff_thres, vector<std::pair<cv::Rect, int> > &box_candidate_vec);
#endif

#if DEBUG_UTIL_TRAFFIC_LIGHTS_RECOGNITION
int traffic_lights_recognition_single_target(const cv::Mat &src_mat, const float diff_thres, 
        vector<std::pair<cv::Rect, int> > &box_candidate_vec, const string img_name);
#else
int traffic_lights_recognition_single_target(const cv::Mat &src_mat, const float diff_thres, 
        vector<std::pair<cv::Rect, int> > &box_candidate_vec);
#endif

// used as recognition interface
#if DEBUG_UTIL_TRAFFIC_LIGHTS_RECOGNITION
int traffic_lights_recognition(const cv::Mat &src_mat, const vector<cv::Rect> &light_target_roi_vec, 
        vector<vector<std::pair<cv::Rect, int> > > &light_recog_result_vec, const string img_name);
#else
int traffic_lights_recognition(const cv::Mat &src_mat, const vector<cv::Rect> &light_target_roi_vec,
        vector<vector<std::pair<cv::Rect, int> > > &light_recog_result_vec);
#endif

int draw_all_result_on_whole_image(const cv::Mat &src_mat, const vector<cv::Rect> &target_roi_vec, const vector<float> &target_score_vec,
        const vector<vector<std::pair<cv::Rect, int> > > &box_candidate_vec, const string result_img_folder, const string image_name, const int class_id);

int draw_recognition_result_on_whole_image(const cv::Mat &src_mat, const vector<cv::Rect> &target_roi_vec, const vector<float> &target_score_vec,
        const vector<vector<std::pair<cv::Rect, int> > > &box_candidate_vec, cv::Mat &draw_mat);

}  // namespace caffe

#endif
