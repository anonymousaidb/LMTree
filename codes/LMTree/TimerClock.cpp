
#include "TimerClock.hpp"


TimerClock::TimerClock() {
    tick();
}


TimerClock::~TimerClock() = default;


void TimerClock::tick() {
    _timer = std::chrono::high_resolution_clock::now();
}


double TimerClock::second() const {
    return static_cast<double>(nanoSec()) * 1e-9;
}


double TimerClock::milliSec() const {
    return static_cast<double>(nanoSec()) * 1e-6;
}


double TimerClock::microSec() const {
    return static_cast<double>(nanoSec()) * 1e-3;
}


long long TimerClock::nanoSec() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - _timer).count();
}
