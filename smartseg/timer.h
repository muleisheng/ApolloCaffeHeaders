#pragma once

namespace smartseg {

inline double current_realtime() {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return tp.tv_sec + tp.tv_nsec * 1e-9;
}

struct Timer {
public:
    Timer() {
        _init_time = current_realtime();
    }
    void hit(const char* func_name, int lineid, const char* tag = "") {
        if (lineid >= (int)_lines.size()) {
            _lines.resize(lineid + 1);
        }
        CHECK(lineid >= 0 && lineid < (int)_lines.size());
        _lines[lineid].func_name = func_name;
        _lines[lineid].total_time += current_realtime();
        _lines[lineid].count++;
        _lines[lineid].tag = tag;
    }
    void print() {
        int last_i = -1;
        for (int i = 0; i < (int)_lines.size(); i++) {
            if (_lines[i].count > 0) {
                if (last_i < 0 || strcmp(_lines[i].func_name, _lines[last_i].func_name) != 0 || _lines[i].count != _lines[last_i].count) {
                    last_i = i;
                }
                printf("Time for %s:%d(%s): %d(count), %.6f(avgtime), %.6f(relative to %d)\n",
                    _lines[i].func_name, i, _lines[i].tag, _lines[i].count, 
                    _lines[i].total_time / _lines[i].count - _init_time,
                    (_lines[i].total_time - _lines[last_i].total_time) / _lines[i].count, last_i);
            }
        }
    }
private:
    struct Item {
        double total_time = 0;
        int count = 0;
        const char* func_name = nullptr;
        const char* tag = nullptr;
    };
    double _init_time;
    std::vector<Item> _lines;
};

}