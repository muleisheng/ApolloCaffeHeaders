#pragma once

namespace smartseg {

class FrameTarget1 {
public:
    void set_objectness_prob_blob(Blob<float>* objectness_prob_blob) {
        _objectness_prob_blob = objectness_prob_blob;
    }
    void set_objectness_target_blob(Blob<float>* objectness_target_blob) {
        _objectness_target_blob = objectness_target_blob;
    }
    void set_objectness_weight_blob(Blob<float>* objectness_weight_blob) {
        _objectness_weight_blob = objectness_weight_blob;
    }
    void set_center_target_blob(Blob<float>* center_target_blob) {
        _center_target_blob = center_target_blob;
    }
    void set_center_weight_blob(Blob<float>* center_weight_blob) {
        _center_weight_blob = center_weight_blob;
    }
    void set_positive_target_blob(Blob<float>* positive_target_blob) {
        _positive_target_blob = positive_target_blob;
    }
    void set_positive_weight_blob(Blob<float>* positive_weight_blob) {
        _positive_weight_blob = positive_weight_blob;
    }
    void set_class_target_blob(Blob<float>* class_target_blob) {
        _class_target_blob = class_target_blob;
    }
    void set_class_weight_blob(Blob<float>* class_weight_blob) {
        _class_weight_blob = class_weight_blob;
    }
    void set_orientation_target_blob(Blob<float>* orientation_target_blob) {
        _orientation_target_blob = orientation_target_blob;
    }
    void set_orientation_weight_blob(Blob<float>* orientation_weight_blob) {
        _orientation_weight_blob = orientation_weight_blob;
    }
    void set_topz_target_blob(Blob<float>* topz_target_blob) {
        _topz_target_blob = topz_target_blob;
    }
    void set_topz_weight_blob(Blob<float>* topz_weight_blob) {
        _topz_weight_blob = topz_weight_blob;
    }
    void init(FrameTargetParameter param, Frame* frame, PlanView* view, Segmentor* segmentor) {
        _param = param;
        _config = &Config::instance();
        _frame = frame;
        _view = view;
        _segmentor = segmentor;

        _rows = _view->rows();
        _cols = _view->cols();
        _grids = _view->grids();
    }
    void reshape() {
        _objectness_target_blob->Reshape(1, 1, _rows, _cols);
        _objectness_weight_blob->Reshape(1, 1, _rows, _cols);
        _center_target_blob->Reshape(1, 2, _rows, _cols);
        _center_weight_blob->Reshape(1, 1, _rows, _cols);
        _positive_target_blob->Reshape(1, 1, _rows, _cols);
        _positive_weight_blob->Reshape(1, 1, _rows, _cols);
        _class_target_blob->Reshape(1, _config->num_classes(), _rows, _cols);
        _class_weight_blob->Reshape(1, 1, _rows, _cols);
        _orientation_target_blob->Reshape(1, 2, _rows, _cols);
        _orientation_weight_blob->Reshape(1, 1, _rows, _cols);
        _topz_target_blob->Reshape(1, 1, _rows, _cols);
        _topz_weight_blob->Reshape(1, 1, _rows, _cols);
    }
    void process(int iter) {
        init_grid_centers();
        init_obstacles();
        compute_class_target();
        compute_objectness_target();
        compute_center_target();
        compute_positive_target(iter);
        compute_orientation_target();
        compute_topz_target();
    }
private:
    FrameTargetParameter _param;
    Config* _config;
    Frame* _frame;
    PlanView* _view;
    Segmentor* _segmentor;

    Blob<float>* _objectness_prob_blob;

    Blob<float>* _objectness_target_blob;
    Blob<float>* _objectness_weight_blob;
    Blob<float>* _center_target_blob;
    Blob<float>* _center_weight_blob;
    Blob<float>* _positive_target_blob;
    Blob<float>* _positive_weight_blob;
    Blob<float>* _class_target_blob;
    Blob<float>* _class_weight_blob;
    Blob<float>* _orientation_target_blob;
    Blob<float>* _orientation_weight_blob;
    Blob<float>* _topz_target_blob;
    Blob<float>* _topz_weight_blob;

    int _rows;
    int _cols;
    int _grids;

    std::vector<Eigen::Vector2f> _grid_centers;
    std::vector<std::vector<int>> _obstacle_grids;

    void init_grid_centers() {
        const LidarPoint* cloud = _frame->cloud.data();
        _grid_centers.resize(_grids);
        for (int row = 0; row < _rows; row++) {
            for (int col = 0; col < _cols; col++) {
                int grid = row * _cols + col;
                if (_view->size(grid) <= 0) {
                    _grid_centers[grid] <<
                        _view->get_x(row), _view->get_y(col);
                } else {
                    if (_param.sample_grid_center()) {
                        int i = *(_view->begin(grid) + local_random_engine()() % _view->size(grid));
                        _grid_centers[grid] << cloud[i].x, cloud[i].y;
                    } else {
                        Eigen::Vector2d center;
                        center << 0.0f, 0.0f;
                        for (int i : (*_view)(grid)) {
                            center(0) += cloud[i].x;
                            center(1) += cloud[i].y;
                        }
                        _grid_centers[grid] = (center / _view->size(grid)).cast<float>();
                    }
                }
            }
        }
    }
    std::vector<int> convering_grids(const ObstacleLabel& obstacle) {
        thread_local std::vector<int> grid_list;
        grid_list.clear();

        double max_dis = std::hypot(obstacle.size(0) * 0.5, obstacle.size(1) * 0.5);
        int min_row = std::max(_view->get_row(obstacle.center(0) - max_dis), 0);
        int max_row = std::min(_view->get_row(obstacle.center(0) + max_dis), _rows - 1);
        int min_col = std::max(_view->get_col(obstacle.center(1) - max_dis), 0);
        int max_col = std::min(_view->get_col(obstacle.center(1) + max_dis), _cols - 1);

        Eigen::Affine2d transform{
            (Eigen::Translation<double, 2>(obstacle.center(0), obstacle.center(1)) *
            Eigen::Rotation2D<double>(obstacle.yaw) *
            Eigen::Scaling(obstacle.size(0) * 0.5, obstacle.size(1) * 0.5)).inverse()};

        for (int row = min_row; row <= max_row; row++) {
            for (int col = min_col; col <= max_col; col++) {
                CHECK(_view->valid_row_col(row, col));
                int grid = row * _cols + col;
                Eigen::Vector2d center = _grid_centers[grid].cast<double>();
                center = transform * center;
                if (center(0) >= -1.0 && center(0) <= 1.0 &&
                    center(1) >= -1.0 && center(1) <= 1.0) {
                    grid_list.push_back(grid);
                }
            }
        }

        return grid_list;
    }
    void init_obstacles() {
        const std::vector<ObstacleLabel>& obstacles = _frame->obstacles;
        _obstacle_grids.resize(obstacles.size());
        for (int i = 0; i < (int)obstacles.size(); i++) {
            _obstacle_grids[i] = convering_grids(obstacles[i]);
        }
    }
    void compute_objectness_target() {
        const float* prob_data = _objectness_prob_blob->cpu_data();
        const float* class_weight_data = _class_weight_blob->cpu_data();
        float* target_data = _objectness_target_blob->mutable_cpu_data();
        float* weight_data = _objectness_weight_blob->mutable_cpu_data();

        const std::vector<ObstacleLabel>& obstacles = _frame->obstacles;
        const std::vector<Segment>& segments = _segmentor->segments();

        caffe::caffe_set(_grids, 0.0f, target_data);
        caffe::caffe_set(_grids, 0.0f, weight_data);

        for (int i = 0; i < (int)obstacles.size(); i++) {
            for (int grid : _obstacle_grids[i]) {
                target_data[grid] = 1;
            }
        }

        bool equal_objectness_weight = _param.equal_objectness_weight();
        for (int i = 0; i < (int)segments.size(); i++) {
            double tot_class_weight = 0;
            double tot_intersect = 0;
            double tot_union = 0;
            for (int grid : segments[i].potential_grids) {
                if (target_data[grid] > 0) {
                    tot_class_weight += class_weight_data[grid];
                    tot_intersect += prob_data[grid];
                    tot_union++;
                } else {
                    tot_union += prob_data[grid];
                }
            }
            if (tot_class_weight > 0) {
                for (int grid : segments[i].potential_grids) {
                    if (target_data[grid] > 0 || equal_objectness_weight) {
                        weight_data[grid] = tot_class_weight / tot_union;
                    } else {
                        weight_data[grid] = tot_class_weight * tot_intersect / (tot_union * tot_union);
                    }
                }
            }
        }
    }
    void compute_center_target() {
        float* target_data = _center_target_blob->mutable_cpu_data();
        float* weight_data = _center_weight_blob->mutable_cpu_data();

        const std::vector<ObstacleLabel>& obstacles = _frame->obstacles;

        caffe::caffe_set(_grids * 2, 0.0f, target_data);
        caffe::caffe_set(_grids, 0.0f, weight_data);

        for (int i = 0; i < (int)obstacles.size(); i++) {
            if (!obstacles[i].train_center) {
                continue;
            }
            Eigen::Vector2f obstacle_center = obstacles[i].center.head<2>();
            if (_param.point_center()) {
                obstacle_center.setZero();
                int count = 0;
                for (int grid : _obstacle_grids[i]) {
                    if (_view->size(grid) > 0) {
                        int row = grid / _cols;
                        int col = grid % _cols;
                        obstacle_center += (Eigen::Vector2f() << _view->get_x(row), _view->get_y(col)).finished();
                        count++;
                    }
                }
                if (count == 0) {
                    continue;
                }
                obstacle_center /= count;
            }
            float weight = 1.0 / _obstacle_grids[i].size();
            for (int grid : _obstacle_grids[i]) {
                int row = grid / _cols;
                int col = grid % _cols;
                Eigen::Vector2f center;
                center << obstacle_center(0) - _view->get_x(row), obstacle_center(1) - _view->get_y(col);
                if (_param.max_center_norm() > 0) {
                    float norm = std::hypot(center(0), center(1));
                    if (norm > _param.max_center_norm()) {
                        center *= _param.max_center_norm() / norm;
                    }
                }
                target_data[grid] = center(0);
                target_data[grid + _grids] = center(1);
                weight_data[grid] = weight;
            }
        }
    }
    void compute_positive_target(int iter) {
        float* target_data = _positive_target_blob->mutable_cpu_data();
        float* weight_data = _positive_weight_blob->mutable_cpu_data();

        const std::vector<ObstacleLabel>& obstacles = _frame->obstacles;
        const std::vector<Segment>& segments = _segmentor->segments();

        caffe::caffe_set(_grids, 0.0f, target_data);
        caffe::caffe_set(_grids, 0.0f, weight_data);

        if (iter < _param.startup_iter()) {
            return;
        }

        for (int i = 0; i < (int)obstacles.size(); i++) {
            for (int grid : _obstacle_grids[i]) {
                target_data[grid] = 1;
            }
        }

        for (int i = 0; i < (int)segments.size(); i++) {
            float weight = 1.0 / segments[i].grids.size();
            for (int grid : segments[i].grids) {
                weight_data[grid] = weight;
            }
        }
    }
    void compute_class_target() {
        int num_classes = _config->num_classes();
        float* target_data = _class_target_blob->mutable_cpu_data();
        float* weight_data = _class_weight_blob->mutable_cpu_data();

        const std::vector<ObstacleLabel>& obstacles = _frame->obstacles;

        caffe::caffe_set(_grids * num_classes, 0.0f, target_data);
        caffe::caffe_set(_grids, 0.0f, weight_data);

        for (int i = 0; i < (int)obstacles.size(); i++) {
            int class_id = obstacles[i].class_id;
            CHECK(class_id >= -1 && class_id < num_classes);
            if (class_id < 0) {
                continue;
            }
            int count = 0;
            for (int grid : _obstacle_grids[i]) {
                target_data[class_id * _grids + grid] = 1;
                if (_view->size(grid) > 0) {
                    count++;
                }
            }
            float weight = 1.0 / count;
            for (int grid : _obstacle_grids[i]) {
                if (_view->size(grid) > 0) {
                    weight_data[grid] = weight;
                }
            }
        }
    }
    void compute_orientation_target() {
        float* target_data = _orientation_target_blob->mutable_cpu_data();
        float* weight_data = _orientation_weight_blob->mutable_cpu_data();
        const float* class_weight_data = _class_weight_blob->cpu_data();

        const std::vector<ObstacleLabel>& obstacles = _frame->obstacles;

        caffe::caffe_set(_grids * 2, 0.0f, target_data);
        std::copy(class_weight_data, class_weight_data + _grids, weight_data);

        for (int i = 0; i < (int)obstacles.size(); i++) {
            if (!obstacles[i].train_orientation) {
                continue;
            }
            float yaw = obstacles[i].yaw * 2;
            float cos_yaw = cos(yaw);
            float sin_yaw = sin(yaw);
            for (int grid : _obstacle_grids[i]) {
                target_data[grid] = cos_yaw;
                target_data[grid + _grids] = sin_yaw;
            }
        }
    }
    void compute_topz_target() {
        float* target_data = _topz_target_blob->mutable_cpu_data();
        float* weight_data = _topz_weight_blob->mutable_cpu_data();
        const float* class_weight_data = _class_weight_blob->cpu_data();

        const std::vector<ObstacleLabel>& obstacles = _frame->obstacles;

        caffe::caffe_set(_grids, 0.0f, target_data);
        std::copy(class_weight_data, class_weight_data + _grids, weight_data);

        for (int i = 0; i < (int)obstacles.size(); i++) {
            float topz = obstacles[i].center(2) + obstacles[i].size(2) * 0.5;
            for (int grid : _obstacle_grids[i]) {
                target_data[grid] = topz;
            }
        }
    }
};

}