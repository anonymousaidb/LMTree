

#ifndef LOCALMODEL_H
#define LOCALMODEL_H


#include<vector>
#include<iostream>
#include<fstream>
#include <chrono>
#include<string>
#include<algorithm>
#include"point.h"
#include"query.h"

class LocalModel {
public:
    std::vector<Point> local_data_;
    size_t point_count_{ 0 };

    LocalModel(std::vector<Point>& data) : local_data_(data), point_count_(data.size()) {}
    LocalModel(std::vector<Point>::iterator& beg, std::vector<Point>::iterator& end) {
        local_data_.assign(beg, end);
        point_count_ = local_data_.size();
    }
    LocalModel() {}

    void FilterPointsForQuery(const Query& query, std::vector<Point>& result_vec) const {
        bool use_circle =
#ifdef QUERY_HAS_CENTER_RADIUS
            true;
#else
            false;
#endif

        for (const auto& pnt : local_data_) {
            bool in_mbr = true;
            for (size_t i = 0; i < Constants::DIM; ++i) {
                if (pnt.elements_[i] < query.low_.elements_[i] || pnt.elements_[i] >= query.high_.elements_[i]) {
                    in_mbr = false; break;
                }
            }
            if (!in_mbr) continue;

            if (use_circle) {
                double dist2 = 0.0, r2 = query.radius_ * query.radius_;
                for (size_t i = 0; i < Constants::DIM; ++i) {
                    double d = pnt.elements_[i] - query.center_.elements_[i];
                    dist2 += d * d;
                    if (dist2 > r2) { in_mbr = false; break; }
                }
                if (!in_mbr) continue;
            }
            result_vec.push_back(pnt);
        }
    }

};




#endif