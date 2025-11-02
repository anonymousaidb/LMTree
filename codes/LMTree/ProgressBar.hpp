#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>

class ProgressBar {
private:
    int bar_width;
    std::string start;
    std::string fill;
    std::string lead;
    std::string remainder;
    std::string end;
    double last_progress = 0.0;
    std::chrono::steady_clock::time_point last_time;

public:
    explicit ProgressBar(int width = 50, std::string start = "[", std::string fill = "=",
                         std::string lead = ">", std::string remainder = " ",
                         std::string end = "]");
    ~ProgressBar();
    void update(double step, const std::string& text);
};


#endif
