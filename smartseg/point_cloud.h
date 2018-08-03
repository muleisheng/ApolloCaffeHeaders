#pragma once

namespace smartseg {

struct alignas(16) LidarPoint {
    union {
        float data[3];
        struct {
            float x;
            float y;
            float z;
        };
    };
    uint8_t intensity;

    Eigen::Vector3f& as_eigen() {
        return *(Eigen::Vector3f*)data;
    }
    const Eigen::Vector3f& as_eigen() const {
        return *(const Eigen::Vector3f*)data;
    }
};

template<class T>
void assign_point_cloud(std::vector<LidarPoint>& cloud, size_t n, const T* pcd) {
    cloud.resize(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        const T& sour = pcd[i];
        LidarPoint& dest = cloud[i];
        dest.x = sour.x;
        dest.y = sour.y;
        dest.z = sour.z;
        dest.intensity = sour.intensity;
    }
}

void load_point_cloud(const std::string& path, std::vector<LidarPoint>& cloud);

}
