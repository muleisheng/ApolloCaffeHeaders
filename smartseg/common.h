#pragma once

namespace smartseg {

inline std::vector<std::string> get_lines_from_file(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream infile(path.c_str());
    CHECK(infile) << "File not found: " << path;
    std::string line;
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    return lines;
}

template<class Dtype>
inline void put_pointer_into_blob(caffe::Blob<Dtype>* blob, void* ptr) {
    blob->Reshape(1, sizeof(void*) / sizeof(Dtype), 1, 1);
    *((void**)blob->mutable_cpu_data()) = ptr;
}

template<class Dtype>
inline void* get_pointer_from_blob(caffe::Blob<Dtype>* blob) {
    CHECK(blob->count() == sizeof(void*) / sizeof(Dtype));
    return *((void**)blob->cpu_data());
}

}