#pragma once
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

namespace utility {

class Timer {

public:
    Timer() { reset(); }

    void reset() __attribute__((noinline)) {
        elapsed_usec_ = 0;
        memset(&start_, 0, sizeof(start_));
        memset(&stop_, 0, sizeof(stop_));
        start();
    }

    void start() __attribute__((noinline)) { gettimeofday(&start_, 0); }

    void stop() __attribute__((noinline)) {
        if (start_.tv_sec != 0 || start_.tv_usec != 0) {
            gettimeofday(&stop_, 0);
            elapsed_usec_ += (stop_.tv_sec - start_.tv_sec) * 1e6;
            elapsed_usec_ += stop_.tv_usec - start_.tv_usec;
        }
        memset(&start_, 0, sizeof(start_));
        memset(&stop_, 0, sizeof(stop_));
    }

    void pause() __attribute__((noinline)) { stop(); }

    double elapsed_ms() __attribute__((noinline)) {
        stop();
        start();
        return double(elapsed_usec_)/1000;
    }

    double total_ms() __attribute__((noinline)) { return elapsed_ms(); }

private:
    int64_t elapsed_usec_;
    struct timeval start_, stop_;
};

}  // namespace utility
