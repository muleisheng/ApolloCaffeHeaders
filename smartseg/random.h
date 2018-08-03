#pragma once

namespace smartseg {

inline std::default_random_engine& local_random_engine() {
    struct engine_wrapper_t {
        std::default_random_engine engine;
        engine_wrapper_t() {
            static std::random_device rd;
            static std::mutex mutex;
            mutex.lock();
            std::seed_seq sseq = {rd(), rd(), rd(), rd()};
            mutex.unlock();
            engine.seed(sseq);
        }
    };
    thread_local engine_wrapper_t r;
    return r.engine;
}

template<class T = double>
std::uniform_real_distribution<T>& local_uniform_real_distribution() {
    thread_local std::uniform_real_distribution<T> distr;
    DCHECK(distr.a() == 0.0 && distr.b() == 1.0);
    return distr;
}

template<class T = double>
T uniform_real() {
    return local_uniform_real_distribution<T>()(local_random_engine());
}

template<class T = double>
T uniform_real(T a, T b) {
    if (a == b) {
        return a;
    }
    return (T)(a + uniform_real<T>() * (b - a));
}

}