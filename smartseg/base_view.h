#pragma once

namespace smartseg {

class BaseView {
public:
    struct Grid {
        const int* begin_ptr;
        const int* end_ptr;

        const int* begin() const {
            return begin_ptr;
        }
        const int* end() const {
            return end_ptr;
        }
    };
    virtual ~BaseView() {
    }
    int rows() const {
       return _rows;
    }
    int cols() const {
        return _cols;
    }
    int grids() const {
        return _grids;
    }
    bool valid_row(int row) const {
        return row >= 0 && row < _rows;
    }
    bool valid_col(int col) const {
        return col >= 0 && col < _cols;
    }
    bool valid_row_col(int row, int col) const {
        return valid_row(row) && valid_col(col);
    }
    virtual bool get_row_col(const LidarPoint& pt, int* row, int* col) const = 0;
    const int* begin(int grid) const {
        return _point_index.data() + _starts[grid];
    }
    const int* begin(int row, int col) const {
        if (!valid_row_col(row, col)) {
            return nullptr;
        }
        return begin(row * _cols + col);
    }
    const int* end(int grid) const {
        return _point_index.data() + (_starts[grid] + _counts[grid]);
    }
    const int* end(int row, int col) const {
        if (!valid_row_col(row, col)) {
            return nullptr;
        }
        return end(row * _cols + col);
    }
    int size(int grid) const {
        return _counts[grid];
    }
    int size(int row, int col) const {
        if (!valid_row_col(row, col)) {
            return 0;
        }
        return size(row * _cols + col);
    }
    // This function is for range-for loop inside a grid
    Grid operator()(int grid) const {
        return {begin(grid), end(grid)};
    }
    Grid operator()(int row, int col) const {
        if (!valid_row_col(row, col)) {
            return {nullptr, nullptr};
        }
        return operator()(row * _cols + col);
    }
    const char* point_in_views() const {
        return _pt_in_views.data();
    }
    const Eigen::Vector2i* point_row_cols() const {
        return _pt_row_cols.data();
    }
    const int* point_grids() const {
        return _pt_grids.data();
    }
protected:
    int _rows, _cols, _grids;

    void init(int rows, int cols) {
        _rows = rows;
        _cols = cols;
        _grids = _rows * _cols;
        CHECK_GT(_rows, 0);
        CHECK_GT(_cols, 0);
    }
    void preprocess_points(int num, const LidarPoint* pc) {
        resize_max(_pt_in_views, num);
        resize_max(_pt_row_cols, num);
        resize_max(_pt_grids, num);
        #pragma omp parallel for
        for (int i = 0; i < num; i++) {
            int row, col;
            _pt_in_views[i] = get_row_col(pc[i], &row, &col);
            _pt_row_cols[i] << row, col;
            if (_pt_in_views[i]) {
                _pt_grids[i] = row * _cols + col;
            } else {
                _pt_grids[i] = -1;
            }
        }
    }
    void build_index_without_occlusion(int num, const LidarPoint* pc) {
        _starts.resize(_grids);
        _counts.resize(_grids);
        memset(_counts.data(), 0, sizeof(_counts[0]) * _grids);
        for (int i = 0; i < num; i++) {
            if (_pt_in_views[i]) {
                _counts[_pt_grids[i]]++;
            }
        }

        int counter = 0;
        for (int grid = 0; grid < _grids; grid++) {
            _starts[grid] = counter;
            counter += _counts[grid];
            _counts[grid] = 0;
        }

        resize_max(_point_index, num);
        for (int i = 0; i < num; i++) {
            if (_pt_in_views[i]) {
                int grid = _pt_grids[i];
                _point_index[_starts[grid] + _counts[grid]++] = i;
            }
        }
    }
    void build_index_with_occlusion(int num, const LidarPoint* pc, const float* distances) {
        _starts.resize(_grids);
        _counts.resize(_grids);
        _point_index.resize(_grids);
        memset(_point_index.data(), -1, sizeof(_point_index[0]) * _grids);

        for (int i = 0; i < num; i++) {
            if (_pt_in_views[i]) {
                int grid = _pt_grids[i];
                int j = _point_index[grid];
                if (j < 0 || distances[i] < distances[j]) {
                    _point_index[grid] = i;
                }
                _pt_in_views[i] = false;
            }
        }

        #pragma omp parallel for
        for (int grid = 0; grid < _grids; grid++) {
            _starts[grid] = grid;
            int i = _point_index[grid];
            if (i < 0) {
                _counts[grid] = 0;
            } else {
                _counts[grid] = 1;
                _pt_in_views[i] = true;
            }
        }
    }
private:
    std::vector<char> _pt_in_views;
    std::vector<Eigen::Vector2i> _pt_row_cols;
    std::vector<int> _pt_grids;

    std::vector<int> _starts;
    std::vector<int> _counts;
    std::vector<int> _point_index;

    template<class T>
    void resize_max(std::vector<T>& p, size_t n) {
        p.resize(std::max(n, p.size()));
    }
};

}