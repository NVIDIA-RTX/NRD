#include "Timer.h"

#include <math.h>

#if defined(_WIN32)
#    include <windows.h>
#elif defined(__linux__) || defined(__SCE__) || defined(__APPLE__)
#    include <time.h>
#    ifdef USE_MONOTONIC_TIMER
constexpr clockid_t CLOCKID = CLOCK_MONOTONIC;
#    else
constexpr clockid_t CLOCKID = CLOCK_REALTIME;
#    endif
#else
#    error "Undefined platform"
#endif

#define MY_MIN(a, b) (a < b ? a : b)
#define MY_MAX(a, b) (a > b ? a : b)

inline uint64_t _GetTicks() {
#if defined(_WIN32)
    uint64_t ticks;
    QueryPerformanceCounter((LARGE_INTEGER*)&ticks);
    return ticks;
#elif defined(__linux__) || defined(__SCE__) || defined(__APPLE__)
    struct timespec spec;
    clock_gettime(CLOCKID, &spec);
    return uint64_t(spec.tv_sec) * 1000000000ull + spec.tv_nsec;
#endif
}

nrd::Timer::Timer() {
#if defined(_WIN32)
    uint64_t ticksPerSecond = 1;
    QueryPerformanceFrequency((LARGE_INTEGER*)&ticksPerSecond);

    m_InvTicksPerMs = 1000.0 / ticksPerSecond;
#elif defined(__linux__) || defined(__SCE__) || defined(__APPLE__)
    m_InvTicksPerMs = 1.0 / 1000000.0;
#endif

    SaveCurrentTime();
}

double nrd::Timer::GetTimeStamp() {
    return _GetTicks() * m_InvTicksPerMs;
}

void nrd::Timer::UpdateElapsedTimeSinceLastSave() {
    double ms = (_GetTicks() - m_Time) * m_InvTicksPerMs;
    m_Delta = float(ms);

    float relativeDelta = fabsf(m_Delta - m_SmoothedDelta) / (MY_MIN(m_Delta, m_SmoothedDelta) + 1e-7f);
    float f = relativeDelta / (1.0f + relativeDelta);

    m_SmoothedDelta = m_SmoothedDelta + (m_Delta - m_SmoothedDelta) * MY_MAX(f, 1.0f / 32.0f);
    m_VerySmoothedDelta = m_VerySmoothedDelta + (m_Delta - m_VerySmoothedDelta) * MY_MAX(f, 1.0f / 64.0f);
}

void nrd::Timer::SaveCurrentTime() {
    m_Time = _GetTicks();
}
