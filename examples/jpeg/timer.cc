#include "examples/jpeg/timer.h"

#include <chrono>
#include <sstream>

static int64_t NanoNow() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now() - std::chrono::system_clock::from_time_t(0))
        .count();
}

Timer::Timer() : start_ns_(NanoNow()) {
}

void Timer::Reset() {
    start_ns_ = NanoNow();
}

int64_t Timer::ElapsedMicro() const {
    return (NanoNow() - start_ns_) / 1000;
}

std::string Timer::ToStringMicro() const {
    std::stringstream ss;
    ss << ElapsedMicro() << " μs";
    return ss.str();
}
