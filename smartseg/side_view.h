#pragma once

namespace smartseg {

class SideView : public BaseView {
public:
    void init(const SideViewParameter& param) {
        init(param.range(), param.splits(), param.side());
    }
    void init(float range, int splits, int side) {
        CHECK_EQ(splits % 2, 0);
        BaseView::init(splits / 2, splits);
        CHECK_GT(range, 0);
        _range = range;
        _resolution = _range / _rows;
        _inverse_resolution = _rows / _range;

        const float coeff_table[4][2][2] = {
            {{1, 0}, {0, 1}},
            {{0, -1}, {1, 0}},
            {{-1, 0}, {0, -1}},
            {{0, 1}, {-1, 0}}
        };
        CHECK(side >= 0 && side < 4);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                _coeffs[i][j] = coeff_table[side][i][j];
            }
        }
    }
    bool get_row_col(float x, float y, float z, int* row, int* col) const {
        float new_x = _coeffs[0][0] * x + _coeffs[0][1] * y + (z + 5);
        float new_y = _coeffs[1][0] * x + _coeffs[1][1] * y;
        *row = (int)std::floor(new_x * _inverse_resolution);
        *col = (int)std::floor((new_y + _range) * _inverse_resolution);
        return valid_row_col(*row, *col);
    }
    bool get_row_col(const LidarPoint& pt, int* row, int* col) const final {
        return get_row_col(pt.x, pt.y, pt.z, row, col);
    }
    void set_cloud(const std::vector<LidarPoint>& cloud) {
        int num = cloud.size();
        const LidarPoint* pc = cloud.data();
        _pt_distances.resize(num);
        #pragma omp parallel for
        for (int i = 0; i < num; i++) {
            _pt_distances[i] = -pc[i].z;
        }
        preprocess_points(num, pc);
        build_index_with_occlusion(num, pc, _pt_distances.data());
    }
private:
    float _range;
    float _resolution;
    float _inverse_resolution;
    float _coeffs[2][2];

    std::vector<float> _pt_distances;
};

}