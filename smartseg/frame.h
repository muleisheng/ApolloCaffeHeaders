#pragma once

namespace smartseg {

struct ObstacleLabel {
    alignas(16) Eigen::Vector3f center; // x,y,z
    alignas(16) Eigen::Vector3f size; // length,width,height
    float yaw;
    std::string class_name;
    int class_id;
    bool train_center;
    bool train_orientation;

    void parse(const std::string& line) {
        std::stringstream str(line);
        str >> class_name;
        train_center = true;
        Eigen::Matrix<float, Eigen::Dynamic, 3> mat;
        mat.resize(8, 3);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK(str >> mat(i, j));
            }
        }
        center(0) = (mat(0, 0) + mat(6, 0)) / 2;
        center(1) = (mat(0, 1) + mat(6, 1)) / 2;
        center(2) = (mat(0, 2) + mat(6, 2)) / 2;
        size(0) = (mat.row(0) - mat.row(3)).norm();
        size(1) = (mat.row(0) - mat.row(1)).norm();
        size(2) = (mat.row(0) - mat.row(4)).norm();
        yaw = std::atan2(mat(0, 1) - mat(3, 1), mat(0, 0) - mat(3, 0));

        if (class_name == "cluster") {
            class_name = "pedestrian";
            train_center = false;
        }
        if (class_name == "midMot") {
            class_name = "";
        }
        CHECK(str);

        if (class_name != "") {
            class_id = Config::instance().class_id(class_name);
        } else {
            class_id = -1;
        }

        train_orientation = Config::instance().class_orientation(class_id);
    }
};

struct Frame {
    std::string name;
    std::vector<LidarPoint> cloud;
    std::vector<ObstacleLabel> obstacles;
    alignas(16) Eigen::Vector3d lidar_position;
    double lidar_yaw;

    void load_point_cloud(const std::string& path) {
        smartseg::load_point_cloud(path, cloud);
        name = boost::filesystem::path(path).filename().string();
    }
    template<class T>
    void assign_point_cloud(size_t n, const T* points) {
        smartseg::assign_point_cloud(cloud, n, points);
    }
    void load_obstacles(const std::string& path) {
        obstacles.clear();
        std::ifstream in(path);
        CHECK(in);
        std::string line;
        while (getline(in, line)) {
            if (!line.empty()) {
                ObstacleLabel obs;
                obs.parse(line);
                obstacles.push_back(obs);
            }
        }
        std::shuffle(obstacles.begin(), obstacles.end(), local_random_engine());
    }
    auto cloud_xyz() ->
        Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>, Eigen::Unaligned, Eigen::OuterStride<sizeof(LidarPoint) / sizeof(float)>> {
        return {cloud[0].data, 3, (int)cloud.size()};
    }
    void flip_world() {
        cloud_xyz().topRows<1>() = -cloud_xyz().topRows<1>();
        #pragma omp parallel for
        for (size_t i = 0; i < obstacles.size(); i++) {
            obstacles[i].center(0) = -obstacles[i].center(0);
            obstacles[i].yaw = M_PI - obstacles[i].yaw;
        }
    }
    void translate_world(double x, double y) {
        Eigen::Vector2f offset;
        offset << x, y;
        cloud_xyz().topRows<2>().colwise() += offset;
        #pragma omp parallel for
        for (size_t i = 0; i < obstacles.size(); i++) {
            obstacles[i].center.head<2>() += offset;
        }
    }
    void rotate_world(double yaw) {
        Eigen::Affine2f transform{Eigen::Rotation2D<float>(yaw)};
        cloud_xyz().topRows<2>() = transform * cloud_xyz().topRows<2>();
        #pragma omp parallel for
        for (size_t i = 0; i < obstacles.size(); i++) {
            obstacles[i].center.head<2>() = transform * obstacles[i].center.head<2>();
            obstacles[i].yaw += yaw;
        }
    }
};

}