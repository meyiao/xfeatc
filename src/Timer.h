#pragma once

#include <iostream>
#include <chrono>

class Timer {
public:
    Timer() : now_(std::chrono::steady_clock::now()) {}

    int64_t ElapsedMilliSecs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - now_).count();
    }

    // Passed time in seconds
    double Elapse() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - now_).count() * 1.0e-6;
    }

    void Reset() {
        now_ = std::chrono::steady_clock::now();
    }

    static double TimeSinceEpoch() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() * 1.0e-6;
    }

private:
    std::chrono::steady_clock::time_point now_;


};
