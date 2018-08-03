#pragma once

namespace smartseg {

class Config : public ConfigParameter {
public:
    static Config& instance() {
        static Config conf;
        return conf;
    }
    void init(const ConfigParameter& param) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_inited) {
            return;
        }
        _inited = true;
        (ConfigParameter&)*this = param;
        _num_classes = class_name_size();
        CHECK(class_orientation_size() == _num_classes);
    }
    int num_classes() const {
        return _num_classes;
    }
    int class_id(const std::string& name) const {
        auto it = std::find(class_name().begin(), class_name().end(), name);
        CHECK(it != class_name().end()) << "Unknown class name: " << name;
        return int(it - class_name().begin());
    }
private:
    bool _inited = false;
    std::mutex _mutex;

    int _num_classes;
};

}