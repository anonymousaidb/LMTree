
#ifndef TimerClock_hpp
#define TimerClock_hpp

#include <chrono>

class TimerClock {
    std::chrono::time_point<std::chrono::high_resolution_clock> _timer;

public:
    TimerClock();
    ~TimerClock();

    void tick();

    [[nodiscard]] double second() const;
    [[nodiscard]] double milliSec() const;
    [[nodiscard]] double microSec() const;
    [[nodiscard]] long long nanoSec() const;
};


#endif
