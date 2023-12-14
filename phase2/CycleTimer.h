#ifndef _SYRAH_CYCLE_TIMER_H_
#define _SYRAH_CYCLE_TIMER_H_

#include <chrono>

class CycleTimer {
public:
    typedef std::chrono::high_resolution_clock::time_point SysClock;

    // Return the current CPU time, in terms of clock ticks.
    static SysClock currentTicks() {
        return std::chrono::high_resolution_clock::now();
    }

    // Return the current CPU time, in terms of seconds.
    static double currentSeconds() {
        auto now = currentTicks();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    }

    // Return the conversion from seconds to ticks.
    static double ticksPerSecond() {
        return 1.0 / secondsPerTick();
    }

    // Return the conversion from ticks to seconds.
    static double secondsPerTick() {
        return std::chrono::duration<double, std::ratio<1>>(std::chrono::high_resolution_clock::period()).count();
    }

    // Return the conversion from ticks to milliseconds.
    static double msPerTick() {
        return secondsPerTick() * 1000.0;
    }

private:
    CycleTimer(); // Prevent instantiation
};

#endif // #ifndef _SYRAH_CYCLE_TIMER_H_
