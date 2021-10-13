#ifndef TIMER_H_
#define TIMER_H_

#include <cstdint>
#include <string>
class Timer {
  public:
    Timer();
    void Reset();
    int64_t ElapsedMicro() const;
    std::string ToStringMicro() const;

  private:
    int64_t start_ns_;
};

#endif  // TIMER_H_
