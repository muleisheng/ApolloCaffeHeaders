#pragma once

namespace smartseg {

class PlanView : public BaseView {
public:
    void init(const PlanViewParameter& param) {
        init(param.range(), param.splits());
    }
    void init(float range, int splits) {
        BaseView::init(splits, splits);
        CHECK_GT(range, 0);
        _range = range;
        _resolution = _range * 2 / _rows;
        _inverse_resolution = _rows / (_range * 2);
    }
    float range() const {
        return _range;
    }
    float resolution() const {
        return _resolution;
    }
    float inverse_resolution() const {
        return _inverse_resolution;
    }
    int get_row(float x) const {
        return (int)std::floor((x + _range) * _inverse_resolution);
    }
    int get_col(float y) const {
        return (int)std::floor((y + _range) * _inverse_resolution);
    }
    bool get_row_col(float x, float y, int* row, int* col) const {
        *row = get_row(x);
        *col = get_col(y);
        return valid_row_col(*row, *col);
    }
    bool get_row_col(const LidarPoint& pt, int* row, int* col) const final {
        return get_row_col(pt.x, pt.y, row, col);
    }
    float get_x(int row) const {
        return ((float)row + 0.5f) * _resolution - _range;
    }
    float get_y(int col) const {
        return ((float)col + 0.5f) * _resolution - _range;
    }
    void set_cloud(const std::vector<LidarPoint>& cloud) {
        int num = cloud.size();
        const LidarPoint* pc = cloud.data();
        preprocess_points(num, pc);
        build_index_without_occlusion(num, pc);
    }
private:
    float _range;
    float _resolution;
    float _inverse_resolution;
};

}