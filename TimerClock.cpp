
#include "TimerClock.hpp"

// Constructor
TimerClock::TimerClock() {
    tick();
}

// Destructor
TimerClock::~TimerClock() = default;

// Reset the timer
void TimerClock::tick() {
    _timer = std::chrono::high_resolution_clock::now();
}

// Return elapsed time in seconds
double TimerClock::second() const {
    return static_cast<double>(nanoSec()) * 1e-9;
}

// Return elapsed time in milliseconds
double TimerClock::milliSec() const {
    return static_cast<double>(nanoSec()) * 1e-6;
}

// Return elapsed time in microseconds
double TimerClock::microSec() const {
    return static_cast<double>(nanoSec()) * 1e-3;
}

// Return elapsed time in nanoseconds
long long TimerClock::nanoSec() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - _timer).count();
}
