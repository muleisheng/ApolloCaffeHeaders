#pragma once

namespace smartseg {

class Segmentor2 {
public:
    void set_objectness_prob_blob(Blob<float>* objectness_prob_blob) {
        _objectness_prob_blob = objectness_prob_blob;
    }
    void set_center_pred_blob(Blob<float>* center_pred_blob) {
        _center_pred_blob = center_pred_blob;
    }
    void set_positive_prob_blob(Blob<float>* positive_prob_blob) {
        _positive_prob_blob = positive_prob_blob;
    }
    void set_class_prob_blob(Blob<float>* class_prob_blob) {
        _class_prob_blob = class_prob_blob;
    }
    void set_orientation_pred_blob(Blob<float>* orientation_pred_blob) {
        _orientation_pred_blob = orientation_pred_blob;
    }
    void set_topz_pred_blob(Blob<float>* topz_pred_blob) {
        _topz_pred_blob = topz_pred_blob;
    }
    void init(SegmentorParameter param, Frame* frame, PlanView* view) {
        _param = param;
        _frame = frame;
        _view = view;
        _training_mode = param.training_mode();

        _rows = _view->rows();
        _cols = _view->cols();
        _grids = _view->grids();
    }
    void segment() {
        int num_classes = Config::instance().num_classes();
        int max_points = Config::instance().max_points();
        CHECK((_objectness_prob_blob->shape() == std::vector<int>{1, 1, max_points, 1}));
        CHECK((_center_pred_blob->shape() == std::vector<int>{1, 2, _rows, _cols}));
        CHECK((_positive_prob_blob->shape() == std::vector<int>{1, 1, _rows, _cols}));
        CHECK((_class_prob_blob->shape() == std::vector<int>{1, num_classes, _rows, _cols}));
        CHECK((_orientation_pred_blob->shape() == std::vector<int>{1, 2, _rows, _cols}));
        CHECK((_topz_pred_blob->shape() == std::vector<int>{1, 1, _rows, _cols}));

        do_segment();

        if (!_training_mode) {
            do_classify();
        }
    }
    std::vector<Segment>& segments() {
        return _segments;
    }
private:
    struct Node {
        Node* parent = nullptr;
        char rank = 0;
        char traversed = 0;
        bool is_center = false;
        int segment_id = -1;
        int segment_next_grid = -1;
    };

    SegmentorParameter _param;
    Frame* _frame;
    PlanView* _view;
    bool _training_mode;

    Blob<float>* _objectness_prob_blob;
    Blob<float>* _center_pred_blob;
    Blob<float>* _positive_prob_blob;
    Blob<float>* _class_prob_blob;
    Blob<float>* _orientation_pred_blob;
    Blob<float>* _topz_pred_blob;

    int _rows;
    int _cols;
    int _grids;
    std::vector<Node> _nodes;
    std::vector<char> _is_objects;

    std::vector<Segment> _segments;

    void do_segment() {
        const float* center_pred_data = _center_pred_blob->cpu_data();
        const float* objectness_prob_data = _objectness_prob_blob->cpu_data();
        const int* point_grids = _view->point_grids();

        _nodes.resize(_grids);
        _is_objects.assign(_grids, false);

        #pragma omp parallel for
        for (int i = 0; i < (int)_frame->cloud.size(); i++) {
            int grid = point_grids[i];
            if (grid >= 0) {
                if (_training_mode || objectness_prob_data[i] >= 0.5) {
                    _is_objects[grid] = true;
                }
            }
        }

        #pragma omp parallel for
        for (int row = 0; row < _rows; row++) {
            for (int col = 0; col < _cols; col++) {
                int grid = row * _cols + col;
                Node* node = &_nodes[grid];
                *node = Node();
                Eigen::Vector2f center_pred;
                center_pred << center_pred_data[grid], center_pred_data[grid + _grids];
                int center_row = _view->get_row(_view->get_x(row) + center_pred(0));
                int center_col = _view->get_col(_view->get_y(col) + center_pred(1));
                center_row = std::min(std::max(center_row, 0), _rows - 1);
                center_col = std::min(std::max(center_col, 0), _cols - 1);
                node->parent = &_nodes[center_row * _cols + center_col];
            }
        }

        for (int grid = 0; grid < _grids; grid++) {
            Node* node = &_nodes[grid];
            if (_is_objects[grid] && node->traversed == 0) {
                traverse(node);
            }
        }

        bool merge_diagonal_grids = _param.merge_diagonal_grids();
        for (int row = 0; row < _rows; row++) {
            for (int col = 0; col < _cols; col++) {
                Node* node = &_nodes[row * _cols + col];
                if (!node->is_center) {
                    continue;
                }
                for (int row2 = row - 1; row2 <= row + 1; row2++) {
                    for (int col2 = col - 1; col2 <= col + 1; col2++) {
                        if ((merge_diagonal_grids || row == row2 || col == col2)
                            && _view->valid_row_col(row2, col2)) {
                            Node* node2 = &_nodes[row2 * _cols + col2];
                            if (node2->is_center) {
                                disjoint_set_union(node, node2);
                            }
                        }
                    }
                }
            }
        }

        _segments.clear();
        for (int row = 0; row < _rows; row++) {
            for (int col = 0; col < _cols; col++) {
                int grid = row * _cols + col;
                if (!_is_objects[grid]) {
                    continue;
                }
                Node* node = &_nodes[grid];
                Node* root = disjoint_set_find(node);
                if (root->segment_id < 0) {
                    root->segment_id = (int)_segments.size();
                    _segments.push_back(Segment());
                    _segments.back().center_grid = root - _nodes.data();
                }
                Segment* seg = &_segments[root->segment_id];
                node->segment_next_grid = seg->first_grid;
                seg->first_grid = grid;
            }
        }

        const float* topz_pred_data = _topz_pred_blob->cpu_data();
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)_segments.size(); i++) {
            Segment* seg = &_segments[i];
            extract_segment(seg, objectness_prob_data, topz_pred_data);
        }
    }
    static void traverse(Node* x) {
        thread_local std::vector<Node*> p;
        p.clear();
        while (x->traversed == 0) {
            p.push_back(x);
            x->traversed = 2;
            x = x->parent;
        }
        if (x->traversed == 2) {
            for (int i = (int)p.size() - 1; i >= 0 && p[i] != x; i--) {
                p[i]->is_center = true;
            }
            x->is_center = true;
            x->parent = x;
        }
        for (Node* y : p) {
            y->traversed = 1;
            y->parent = x->parent;
        }
    }
    void extract_segment(Segment* seg, const float* objectness_prob_data, const float* topz_pred_data) {
        thread_local std::vector<int> potential_grids;
        potential_grids.clear();
        Node* node = nullptr;
        for (int grid = seg->first_grid; grid >= 0; grid = node->segment_next_grid) {
            node = &_nodes[grid];
            potential_grids.push_back(grid);
        }
        seg->potential_grids = potential_grids;

        thread_local std::vector<int> grids;
        grids.clear();
        double topz = 0;
        for (int grid : potential_grids) {
            for (int i : (*_view)(grid)) {
                if (objectness_prob_data[i] >= 0.5) {
                    grids.push_back(grid);
                    topz += topz_pred_data[grid];
                    break;
                }
            }
        }
        seg->grids = grids;
        topz /= grids.size();
        seg->topz = topz;

        const LidarPoint* cloud = _frame->cloud.data();
        float topz_threshold = _param.topz_threshold();
        thread_local std::vector<int> points;
        points.clear();
        for (int grid : grids) {
            for (int i : (*_view)(grid)) {
                if (objectness_prob_data[i] >= 0.5 &&
                    (topz_threshold < 0 || cloud[i].z <= topz + topz_threshold)) {
                    points.push_back(i);
                }
            }
        }
        seg->points = points;
    }
    void do_classify() {
        const float* positive_prob_data = _positive_prob_blob->cpu_data();
        int num_classes = Config::instance().num_classes();
        const float* class_prob_data = _class_prob_blob->cpu_data();
        const float* orientation_pred_data = _orientation_pred_blob->cpu_data();

        #pragma omp parallel for schedule(dynamic)
        for (int seg_id = 0; seg_id < (int)_segments.size(); seg_id++) {
            Segment* seg = &_segments[seg_id];

            double positive_prob = 0;

            Eigen::VectorXd class_prob(num_classes);
            class_prob.setZero();

            double cos_yaw = 0;
            double sin_yaw = 0;

            for (int grid : seg->grids) {
                positive_prob += positive_prob_data[grid];
                for (int i = 0; i < num_classes; i++) {
                    class_prob(i) += class_prob_data[i * _grids + grid];
                }
                cos_yaw += orientation_pred_data[grid];
                sin_yaw += orientation_pred_data[grid + _grids];
            }

            positive_prob /= seg->grids.size();
            seg->positive_prob = positive_prob;

            class_prob /= seg->grids.size();
            seg->class_prob = class_prob.cast<float>();

            seg->yaw = atan2(sin_yaw, cos_yaw) / 2;
        }
    }
};

typedef Segmentor2 Segmentor;

}