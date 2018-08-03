#pragma once

namespace smartseg {

template<class T>
class Channel : public boost::noncopyable {
public:
    Channel() {
    }
    explicit Channel(size_t capacity) { // capacity can be zero
        _capacity = std::min(max_capacity(), capacity);
    }
    size_t capacity() {
        return _capacity; // atomic
    }
    void set_capacity(size_t x) { // capacity can be zero
        std::lock_guard<std::mutex> lock(_mutex);
        _capacity = std::min(max_capacity(), x);
        notify();
    }
    bool closed() {
        return _closed; // atomic
    }
    void open() {
        std::lock_guard<std::mutex> lock(_mutex);
        _closed = false;
        notify();
    }
    void close() {
        std::lock_guard<std::mutex> lock(_mutex);
        _closed = true;
        notify();
    }
    size_t size() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _data.size();
    }
    bool empty() {
        std::lock_guard<std::mutex> lock(_mutex);
        return empty_unlocked();
    }
    // blocking operation
    bool get(T& val) {
        return read(1, &val) != 0;
    }
    // blocking operation
    // returns 0 if the channel is closed and empty
    size_t read(size_t n, T* p) {
        if (n == 0) {
            return 0;
        }
        std::unique_lock<std::mutex> lock(_mutex);
        size_t finished = read(n, p, lock);
        notify();
        return finished;
    }
    // blocking operation
    bool put(T&& val) {
        return write_move(1, &val) != 0;
    }
    // blocking operation
    bool put(const T& val) {
        return write(1, &val) != 0;
    }
    // blocking operation
    // returns value less than n if the channel is closed
    size_t write(size_t n, const T* p) {
        if (n == 0) {
            return 0;
        }
        std::unique_lock<std::mutex> lock(_mutex);
        size_t finished = write(n, p, lock);
        notify();
        return finished;
    }
    // write_move() will clear original contents of p
    size_t write_move(size_t n, T* p) {
        if (n == 0) {
            return 0;
        }
        std::unique_lock<std::mutex> lock(_mutex);
        size_t finished = write_move(n, p, lock);
        notify();
        return finished;
    }
    size_t read(std::vector<T>& p, size_t n) {
        p.resize(n);
        size_t finished = read(n, &p[0]);
        p.resize(finished);
        return finished;
    }
    size_t write(const std::vector<T>& p) {
        return write(p.size(), &p[0]);
    }
    size_t write(std::vector<T>&& p) {
        return write_move(p.size(), &p[0]);
    }
private:
    size_t _capacity = max_capacity();
    bool _closed = false;

    std::mutex _mutex;
    std::deque<T> _data;
    size_t _reading_count = 0;
    int _empty_waiters = 0;
    int _full_waiters = 0;
    std::condition_variable _empty_cond;
    std::condition_variable _full_cond;

    static constexpr size_t max_capacity() {
        return std::numeric_limits<size_t>::max() / 2;
    }
    void notify() {
        if (_empty_waiters != 0 && (!empty_unlocked() || _closed)) {
            _empty_cond.notify_one();
        }
        if (_full_waiters != 0 && (!full_unlocked() || _closed)) {
            _full_cond.notify_one();
        }
    }
    bool empty_unlocked() {
        return _data.empty();
    }
    bool full_unlocked() {
        return _data.size() >= _capacity + _reading_count;
    }
    bool wait_for_read(std::unique_lock<std::mutex>& lock) {
        while (unlikely(empty_unlocked() && !_closed)) {
            if (_full_waiters != 0) {
                _full_cond.notify_one();
            }
            _empty_waiters++;
            _empty_cond.wait(lock);
            _empty_waiters--;
        }
        return !empty_unlocked();
    }
    bool wait_for_write(std::unique_lock<std::mutex>& lock) {
        while (unlikely(full_unlocked() && !_closed)) {
            if (_empty_waiters != 0) {
                _empty_cond.notify_one();
            }
            _full_waiters++;
            _full_cond.wait(lock);
            _full_waiters--;
        }
        return !_closed;
    }
    size_t read(size_t n, T* p, std::unique_lock<std::mutex>& lock) {
        size_t finished = 0;
        CHECK(n <= max_capacity() - _reading_count);
        _reading_count += n;
        while (finished < n && wait_for_read(lock)) {
            size_t m = std::min(n - finished, _data.size());
            for (size_t i = 0; i < m; i++) {
                p[finished++] = std::move(_data.front());
                _data.pop_front();
            }
            _reading_count -= m;
        }
        _reading_count -= n - finished;
        return finished;
    }
    size_t write(size_t n, const T* p, std::unique_lock<std::mutex>& lock) {
        size_t finished = 0;
        while (finished < n && wait_for_write(lock)) {
            size_t m = std::min(n - finished, _capacity + _reading_count - _data.size());
            for (size_t i = 0; i < m; i++) {
                _data.push_back(p[finished++]);
            }
        }
        return finished;
    }
    size_t write_move(size_t n, T* p, std::unique_lock<std::mutex>& lock) {
        size_t finished = 0;
        while (finished < n && wait_for_write(lock)) {
            size_t m = std::min(n - finished, _capacity + _reading_count - _data.size());
            for (size_t i = 0; i < m; i++) {
                _data.push_back(std::move(p[finished++]));
            }
        }
        return finished;
    }
};

}
