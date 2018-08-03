#pragma once

namespace smartseg {

class HorizontalView : public BaseView {
public:
    void init(const HorizontalViewParameter& param) {
        init(param.lower_pitch(), param.upper_pitch(), param.rows(), param.cols());
    }
    void init(float lower_pitch, float upper_pitch, int rows, int cols) {
        BaseView::init(rows, cols);
        CHECK_LT(lower_pitch, upper_pitch);
        _lower_pitch = lower_pitch;
        _upper_pitch = upper_pitch;
        _yaw_resolution = 2 * M_PI / cols;
        _inverse_yaw_resolution = cols / (2 * M_PI);
        _pitch_resolution = (upper_pitch - lower_pitch) / rows;
        _inverse_pitch_resolution = rows / (upper_pitch - lower_pitch);
    }
    bool get_row_col(float x, float y, float z, int* row, int* col) const {
        float yaw = std::atan2(y, x);
        float pitch = z / std::hypot(x, y);
        *row = (int)std::floor((pitch - _lower_pitch) * _inverse_pitch_resolution);
        *col = (int)std::floor((yaw + (float)M_PI) * _inverse_yaw_resolution);
        if (*col == -1) {
            *col = _cols - 1;
        } else if (*col == _cols) {
            *col = 0;
        }
        return valid_row_col(*row, *col);
    }
    bool get_row_col(const LidarPoint& pt, int* row, int* col) const final {
        return get_row_col(pt.x, pt.y, pt.z, row, col);
    }
    float get_yaw(int col) const {
        return ((float)col + 0.5f) * _yaw_resolution - M_PI;
    }
    float get_pitch(int row) const {
        return ((float)row + 0.5f) * _pitch_resolution + _lower_pitch;
    }
    const float* point_distances() const {
        return _pt_distances.data();
    }
    void set_cloud(const std::vector<LidarPoint>& cloud) {
        int num = cloud.size();
        const LidarPoint* pc = cloud.data();
        _pt_distances.resize(num);
        #pragma omp parallel for
        for (int i = 0; i < num; i++) {
            _pt_distances[i] = std::hypot(pc[i].x, pc[i].y);
        }
        preprocess_points(num, pc);
        build_index_with_occlusion(num, pc, _pt_distances.data());
    }
private:
    float _lower_pitch;
    float _upper_pitch;
    float _yaw_resolution;
    float _inverse_yaw_resolution;
    float _pitch_resolution;
    float _inverse_pitch_resolution;

    std::vector<float> _pt_distances;
};

}