#pragma once

namespace smartseg {

struct FrameSequence {
    std::vector<Frame> frames;

    void load(
            const std::string& pcd_path,
            const std::string& label_path,
            const std::string& seq_path,
            const std::string& dataset_name,
            int num_frames,
            int step) {
        std::string dataset_filename = boost::filesystem::path(dataset_name).filename().string();
        std::string bag_name;
        int id;
        parse_dataset_name(dataset_filename, &bag_name, &id);
        std::map<int, std::string> pose_map = load_pose_file(str(boost::format("%s/%s/pose.txt") % seq_path % bag_name));

        frames.clear();
        frames.resize(num_frames);
        frames[0].load_point_cloud(pcd_path + "/" + dataset_name);
        if (!label_path.empty()) {
            frames[0].load_obstacles(label_path + "/" + dataset_name + ".txt");
        }
        bool frame0_has_pose = pose_map.find(id) != pose_map.end();
        if (frame0_has_pose) {
            parse_pose(&frames[0], pose_map[id]);
        } else {
            LOG(WARNING) << "Frame 0 has no pose: " << dataset_name;
            frames[0].lidar_position.setZero();
            frames[0].lidar_yaw = 0;
        }

        for (int i = 1; i < num_frames; i++) {
            id -= step;
            std::string path = str(boost::format("%s/%s/%d.pcd") % seq_path % bag_name % id);
            bool has_pcd = boost::filesystem::exists(path);
            bool has_pose = pose_map.find(id) != pose_map.end();
            if (!frame0_has_pose) {
            } else if (!has_pcd) {
                LOG(WARNING) << "Frame " << i << " is missing: " << path;
            } else if (!has_pose) {
                LOG(WARNING) << "Frame " << i << " has no pose: " << dataset_name;
            }
            if (!frame0_has_pose || !has_pcd || !has_pose) {
                frames[i] = frames[i - 1];
                continue;
            }
            frames[i].load_point_cloud(path);
            parse_pose(&frames[i], pose_map[id]);
        }
        for (int i = 0; i < num_frames; i++) {
            frames[i].name = str(boost::format("%s#%02d") % dataset_filename % i);
        }
    }
    void align_frames() {
        for (int i = 1; i < (int)frames.size(); i++) {
            Eigen::Vector2d offset = (frames[i].lidar_position - frames[0].lidar_position).head<2>();
            offset = Eigen::Rotation2D<double>(-frames[i].lidar_yaw) * offset;
            frames[i].translate_world(offset(0), offset(1));
            frames[i].rotate_world(frames[i].lidar_yaw - frames[0].lidar_yaw);
        }
    }
    void flip_world() {
        for (int i = 0; i < (int)frames.size(); i++) {
            frames[i].flip_world();
        }
    }
    void rotate_world(double yaw) {
        for (int i = 0; i < (int)frames.size(); i++) {
            frames[i].rotate_world(yaw);
        }
    }
private:
    void parse_dataset_name(const std::string& dataset_name, std::string* bag_name, int* id) {
        std::vector<std::string> parts;
        boost::split(parts, dataset_name, boost::is_any_of("_."));
        *bag_name = str(boost::format("%s_12_%s_%s") % parts[0] % parts[2] % parts[3]);
        CHECK(sscanf(parts[4].c_str(), "%d", id) == 1);
    }
    std::map<int, std::string> load_pose_file(const std::string& path) {
        if (!boost::filesystem::exists(path)) {
            LOG(WARNING) << "Pose file is missing: " << path;
            return {};
        }
        std::vector<std::string> lines = get_lines_from_file(path);
        std::map<int, std::string> pose_map;
        for (const std::string& line : lines) {
            int id;
            CHECK(sscanf(line.c_str(), "%d", &id) == 1);
            pose_map.insert({id, line});
        }
        return pose_map;
    }
    void parse_pose(Frame* frame, const std::string& line) {
        Eigen::Vector3d pos;
        Eigen::Quaternion<double> quat;
        CHECK(sscanf(line.c_str(), "%*d %*f %lf %lf %lf %lf %lf %lf %lf", 
            &pos(0), &pos(1), &pos(2), &quat.x(), &quat.y(), &quat.z(), &quat.w()) == 7);
        Eigen::Vector3d vec;
        vec << 1, 0, 0;
        vec = quat * vec;
        frame->lidar_position = pos;
        frame->lidar_yaw = atan2(vec(1), vec(0));
    }
};

}