#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>
#include "Parameters.hpp"
#include "ProgressBar.hpp"

ProgressBar::ProgressBar(const int width, std::string start, std::string fill,
                         std::string lead, std::string remainder, std::string end)
        : bar_width(width), start(std::move(start)), fill(std::move(fill)),
          lead(std::move(lead)), remainder(std::move(remainder)), end(std::move(end)) {
    last_time = std::chrono::steady_clock::now();
}

ProgressBar::~ProgressBar()
{
    std::cout << std::endl;
}

void ProgressBar::update(const double step, const std::string& text) {
    const auto current_time = std::chrono::steady_clock::now();
    const auto elapsed_time = std::chrono::duration<double>(current_time - last_time).count();
    last_time = current_time;

    double progress_difference = step - last_progress;
    last_progress = step;
    if (progress_difference <= 0.0) {
        progress_difference = 1e-20;
    }


    const double remaining_time = elapsed_time * (1.0 - last_progress) / progress_difference;

    const int filled_length = static_cast<int>(last_progress * bar_width);
    std::cout << "\r" << start;

    for (int i = 0; i < bar_width; ++i) {
        if (i < filled_length)
            std::cout << fill;
        else if (i == filled_length)
            std::cout << lead;
        else
            std::cout << remainder;
    }


    std::cout << end << " "
              << std::fixed << std::setprecision(2)
              << last_progress * 100.0 << "% " << text;

    int remaining_seconds = static_cast<int>(remaining_time);
    if (false) {
        std::cout << " | Time left: " << remaining_seconds << "s" << std::flush;
    }else {
        const int hours = remaining_seconds / 3600;
        remaining_seconds %= 3600;
        const int minutes = remaining_seconds / 60;
        const int seconds = remaining_seconds % 60;

        if (hours > 0) {
            std::cout << " | Time left: " << hours << "h " << minutes << "m " << seconds << "s" << std::flush;
        } else if (minutes > 0) {
            std::cout << " | Time left: " << minutes << "m " << seconds << "s" << std::flush;
        } else {
            std::cout << " | Time left: " << seconds << "s" << std::flush;
        }
    }
}
