#ifndef FLOODLITE_H
#define FLOODLITE_H

#include<cstdint>
#include<cmath>
#include<vector>
#include<algorithm>
#include<iostream>
#include<limits>
#include <chrono>
#include<list>
#include<cassert>
#include <array>
#include <numeric>

#include <queue>


#include"point.h"
#include"query.h"
#include"sort_tools.h"
#include"local_model.h"
#include <random>
#include <queue>


enum class MetricType { L1, L2, LINF };
inline double axis_outside(double x, double lo, double hi) {
    if (x < lo) return lo - x;
    if (x >= hi) return x - hi;
    return 0.0;
}

inline double point_distance(const Point& a, const Point& b, MetricType m) {
    disCount++;
    switch (m) {
    case MetricType::L1: {
        double s = 0.0;
        for (int i = 0; i < Constants::DIM; ++i)
            s += std::abs(a.elements_[i] - b.elements_[i]);
        return s;
    }
    case MetricType::L2: {
        double s2 = 0.0;
        for (int i = 0; i < Constants::DIM; ++i) {
            double d = a.elements_[i] - b.elements_[i];
            s2 += d * d;
        }
        return std::sqrt(s2);
    }
    case MetricType::LINF: {
        double mval = 0.0;
        for (int i = 0; i < Constants::DIM; ++i)
            mval = std::max(mval, std::abs(a.elements_[i] - b.elements_[i]));
        return mval;
    }
    }
    return 0.0;
}

inline double point_to_cell_mindist(const Point& c,
    double col_lo, double col_hi,
    double page_lo, double page_hi,
    int primary_axis, int secondary_axis,
    MetricType m) {
    disCount++;
    const double dx = axis_outside(c.elements_[primary_axis], col_lo, col_hi);
    const double dy = axis_outside(c.elements_[secondary_axis], page_lo, page_hi);
    switch (m) {
    case MetricType::L1:   return dx + dy;
    case MetricType::L2:   return std::sqrt(dx * dx + dy * dy);
    case MetricType::LINF: return std::max(dx, dy);
    }
    return 0.0;
}

inline bool point_within_ball(const Point& p, const Point& c, double radius, MetricType m) {
    return point_distance(p, c, m) <= radius + 1e-15;
}




struct LexiComparatorByOrder {
    const std::array<int, 128>& order;
    int dim;
    LexiComparatorByOrder(const std::array<int, 128>& o, int d) : order(o), dim(d) {}
    template<class P>
    bool operator()(const P& a, const P& b) const {
        for (int i = 0; i < dim; ++i) {
            int ax = order[i];
            if (a.elements_[ax] < b.elements_[ax]) return true;
            if (a.elements_[ax] > b.elements_[ax]) return false;
        }
        return false;
    }
};


inline std::array<int, 128> RandomDimOrder(int dim) {
    std::array<int, 128> order{};
    for (int i = 0; i < dim; ++i) order[i] = i;
    static thread_local std::mt19937_64 rng{ std::random_device{}() };
    std::shuffle(order.begin(), order.begin() + dim, rng);
    return order;
}

class FloodLite
{
public:
    std::array<int, 128> dim_order_{};
    uint32_t num_grid_splits_ = 2;
    uint32_t page_cnt_ = 0;



    std::vector<double> grid_split_;
    std::vector<std::vector<double>> local_model_split_;
    std::vector<std::vector<size_t>> local_model_ids_;
    std::vector<LocalModel> cell_list_;

    MetricType metric_ = MetricType::L2;


    FloodLite(std::vector<Point>& datapoints, std::vector<Query>& queries,
        MetricType metric = MetricType::L2)
        : metric_(metric) {
        std::random_shuffle(queries.begin(), queries.end());
        std::vector<Query> query_sub(queries.begin(),
            queries.begin() + (queries.empty() ? 0 : std::max<size_t>(1, size_t(queries.size() * 0.2))));


        std::random_shuffle(datapoints.begin(), datapoints.end());
        std::vector<Point> data_sub(datapoints.begin(),
            datapoints.begin() + (datapoints.empty() ? 0 : std::max<size_t>(1, size_t(datapoints.size() * 0.2))));


        std::mt19937_64 rng{ std::random_device{}() };
        std::uniform_int_distribution<uint32_t> split_dist(2, 20);


        const int DIM = Constants::DIM;
        uint64_t best_time = std::numeric_limits<uint64_t>::max();
        std::array<int, 128> best_order{};
        uint32_t best_splits = 2;


        uint32_t patience = 3;
        while (patience--) {
            dim_order_ = RandomDimOrder(DIM);
            num_grid_splits_ = split_dist(rng);


            BuildFlood(data_sub);


            auto t0 = std::chrono::high_resolution_clock::now();
            for (auto& q : query_sub) {
                std::vector<size_t> cells; Projection(cells, q);
                std::vector<Point> out; Scan(cells, q, out);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            uint64_t cost = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();


            if (cost < best_time) {
                best_time = cost;
                best_order = dim_order_;
                best_splits = num_grid_splits_;
                patience = 3;
            }
        }


        dim_order_ = best_order;
        num_grid_splits_ = best_splits;
        BuildFlood(datapoints);
    }
    FloodLite(std::vector<Point>& datapoints, const std::string& filename,
        MetricType metric = MetricType::L2)
        : metric_(metric) {
        std::ifstream fin(filename);
        if (!fin) throw std::runtime_error("Cannot open model file");


        bool ok_new = true;
        for (int i = 0; i < Constants::DIM; ++i) {
            if (!(fin >> dim_order_[i])) { ok_new = false; break; }
        }
        if (ok_new) {
            fin >> num_grid_splits_;
        }
        else {
            fin.clear(); fin.seekg(0);
            bool grid_orientation = false;
            fin >> grid_orientation >> num_grid_splits_;
            if (Constants::DIM >= 2) {
                if (!grid_orientation) { dim_order_[0] = 0; dim_order_[1] = 1; }
                else { dim_order_[0] = 1; dim_order_[1] = 0; }
                for (int i = 2; i < Constants::DIM; ++i) dim_order_[i] = i;
            }
            else if (Constants::DIM == 1) {
                dim_order_[0] = 0;
            }
        }
        BuildFlood(datapoints);
    }
    void SaveFlood(const std::string& file) const {
        std::ofstream fout(file);
        for (int i = 0; i < Constants::DIM; ++i) fout << dim_order_[i] << ' ';
        fout << num_grid_splits_;
    }
    void BuildFlood(std::vector<Point>& datapoints) {
        const int DIM = Constants::DIM;
        const int primary_axis = dim_order_[0];
        const int secondary_axis = (DIM >= 2 ? dim_order_[1] : dim_order_[0]);


        grid_split_.clear();
        local_model_split_.clear();
        local_model_ids_.clear();
        cell_list_.clear();
        page_cnt_ = 0;


        local_model_split_.resize(num_grid_splits_);
        local_model_ids_.resize(num_grid_splits_);


        if (datapoints.empty()) return;


        {
            auto order = dim_order_;
            std::sort(datapoints.begin(), datapoints.end(), LexiComparatorByOrder(order, DIM));
        }


        uint32_t col_start = 0;
        for (uint32_t col_id = 0; col_id < num_grid_splits_; ++col_id) {
            uint32_t remaining = uint32_t(datapoints.size()) - col_start;
            uint32_t buckets = num_grid_splits_ - col_id;
            uint32_t take = (remaining + buckets - 1) / buckets;
            uint32_t col_end = std::min<uint32_t>(col_start + take, uint32_t(datapoints.size()));


            double local_low = datapoints[col_start].elements_[primary_axis];
            double local_high = datapoints[col_end - 1].elements_[primary_axis];
            if (col_id == 0) local_low = -std::numeric_limits<double>::infinity();
            if (col_id == num_grid_splits_ - 1) local_high = std::numeric_limits<double>::infinity();
            grid_split_.push_back(local_high + Constants::EPSILON_ERR);



            auto beg = datapoints.begin() + col_start;
            auto end = datapoints.begin() + col_end;
            std::stable_sort(beg, end, [secondary_axis](const Point& a, const Point& b) {
                return a.elements_[secondary_axis] < b.elements_[secondary_axis];
                });


            for (uint32_t lm_beg = col_start; lm_beg < col_end; lm_beg += LOCAL_MODEL_SIZE) {
                uint32_t lm_end = std::min<uint32_t>(lm_beg + uint32_t(LOCAL_MODEL_SIZE), col_end);
                size_t lm_id = cell_list_.size();
                local_model_ids_[col_id].push_back(lm_id);


                cell_list_.push_back(LocalModel());
                ++page_cnt_;
                cell_list_.back().local_data_.assign(datapoints.begin() + lm_beg, datapoints.begin() + lm_end);


                local_model_split_[col_id].push_back(datapoints[lm_end - 1].elements_[secondary_axis] + Constants::EPSILON_ERR);
            }
            col_start = col_end;
        }
    }
    void Projection(std::vector<size_t>& projected_cell_ids, const Query& query) const {
        const int DIM = Constants::DIM;
        const int primary_axis = dim_order_[0];
        const int secondary_axis = (DIM >= 2 ? dim_order_[1] : dim_order_[0]);


        auto first_col = std::upper_bound(grid_split_.begin(), grid_split_.end(), query.low_.elements_[primary_axis]);
        auto last_col = std::upper_bound(grid_split_.begin(), grid_split_.end(), query.high_.elements_[primary_axis]);
        uint32_t col_id_st = uint32_t(first_col - grid_split_.begin());
        uint32_t col_id_last = uint32_t(last_col - grid_split_.begin());
        if (col_id_last >= num_grid_splits_) col_id_last = num_grid_splits_ - 1;


        for (uint32_t col_ix = col_id_st; col_ix <= col_id_last; ++col_ix) {
            auto& page_split = local_model_split_[col_ix];
            if (page_split.empty()) continue;


            auto first_page = std::upper_bound(page_split.begin(), page_split.end(), query.low_.elements_[secondary_axis]);
            if (first_page == page_split.end()) continue;
            uint32_t first_page_id = uint32_t(first_page - page_split.begin());


            auto last_page = std::upper_bound(page_split.begin(), page_split.end(), query.high_.elements_[secondary_axis]);
            uint32_t last_page_id = (last_page == page_split.end() ? uint32_t(page_split.size() - 1)
                : uint32_t(last_page - page_split.begin()));
            for (uint32_t page_ix = first_page_id; page_ix <= last_page_id; ++page_ix) {
                projected_cell_ids.push_back(local_model_ids_[col_ix][page_ix]);
            }
        }
    }
    void Scan(std::vector<size_t>& projected_cell_ids, const Query& query, std::vector<Point>& result_vec) const {
        std::sort(projected_cell_ids.begin(), projected_cell_ids.end());
        projected_cell_ids.erase(std::unique(projected_cell_ids.begin(), projected_cell_ids.end()),
            projected_cell_ids.end());

        const Point  center = InferQueryCenter(query);
        const double radius = queryRadius;

        for (auto c_id : projected_cell_ids) {
            const auto& page = cell_list_[c_id].local_data_;
            for (const auto& p : page) {
                if (!PointWithinMBR(p, query)) continue;

                if (!point_within_ball(p, center, radius, metric_)) continue;

                result_vec.push_back(p);
            }
        }
    }
   
    

    void KNN(const Point& center, size_t k, std::vector<Point>& out) const {
        out.clear();
        if (k == 0 || cell_list_.empty()) return;

        const int DIM = Constants::DIM;
        const int primary_axis = dim_order_[0];
        const int secondary_axis = (DIM >= 2 ? dim_order_[1] : dim_order_[0]);

        std::pair<uint32_t, uint32_t> loc = LocateCell(center);
        uint32_t col0 = loc.first;
        uint32_t page0 = loc.second;

        struct CellItem { double mind; uint32_t col, page; size_t lm_id; };
        struct CellCmp { bool operator()(const CellItem& a, const CellItem& b) const { return a.mind > b.mind; } };
        std::priority_queue<CellItem, std::vector<CellItem>, CellCmp> cellq;

        std::vector<std::vector<uint8_t>> visited(num_grid_splits_);
        for (uint32_t c = 0; c < num_grid_splits_; ++c)
            visited[c].assign(local_model_split_[c].size(), 0);

        struct Hit { double d; const Point* p; };
        struct HitCmp { bool operator()(const Hit& a, const Hit& b) const { return a.d < b.d; } };
        std::priority_queue<Hit, std::vector<Hit>, HitCmp> topk;

        auto push_cell = [&](uint32_t c, uint32_t p) {
            if (c >= num_grid_splits_) return;
            if (p >= local_model_split_[c].size()) return;
            if (visited[c][p]) return;
            visited[c][p] = 1;

            std::pair<double, double> cb = ColumnBounds(c);
            std::pair<double, double> pb = PageBounds(c, p);
            double clo = cb.first, chi = cb.second;
            double plo = pb.first, phi = pb.second;

            double md = point_to_cell_mindist(center, clo, chi, plo, phi,
                primary_axis, secondary_axis, metric_);
            cellq.push(CellItem{ md, c, p, local_model_ids_[c][p] });
        };

        auto kth_dist = [&]()->double {
            return topk.size() < k ? std::numeric_limits<double>::infinity() : topk.top().d;
        };

        push_cell(col0, page0);

        while (!cellq.empty()) {
            CellItem cur = cellq.top(); cellq.pop();

            if (topk.size() >= k && cur.mind >= kth_dist()) break;

            const std::vector<Point>& page = cell_list_[cur.lm_id].local_data_;
            for (size_t i = 0; i < page.size(); ++i) {
                const Point& p = page[i];
                double d = point_distance(p, center, metric_);
                if (topk.size() < k) {
                    topk.push(Hit{ d, &p });
                }
                else if (d < topk.top().d) {
                    topk.pop();
                    topk.push(Hit{ d, &p });
                }
            }

            if (cur.page > 0)                           push_cell(cur.col, cur.page - 1);
            if (cur.page + 1 < visited[cur.col].size()) push_cell(cur.col, cur.page + 1);
            if (cur.col > 0) {
                if (cur.page < visited[cur.col - 1].size()) push_cell(cur.col - 1, cur.page);
            }
            if (cur.col + 1 < num_grid_splits_) {
                if (cur.page < visited[cur.col + 1].size()) push_cell(cur.col + 1, cur.page);
            }
        }

        std::vector<Hit> hits;
        hits.reserve(topk.size());
        while (!topk.empty()) { hits.push_back(topk.top()); topk.pop(); }
        std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.d < b.d; });
        out.reserve(hits.size());
        for (size_t i = 0; i < hits.size(); ++i) out.push_back(*hits[i].p);
    }

    void KNNBatchRounds(const std::vector<Query>& queries,
        const std::vector<size_t>& k_list,
        std::vector<std::vector<std::vector<Point>>>& results) const {
        results.clear();
        results.resize(k_list.size());
        for (size_t r = 0; r < k_list.size(); ++r) {
            size_t k = k_list[r];
            auto& round_out = results[r];
            round_out.resize(queries.size());
            for (size_t i = 0; i < queries.size(); ++i) {
                KNN(queries[i].center_, k, round_out[i]);
            }
        }
    }

    bool Insert(const Point& p) {
        uint32_t col = 0, page = 0; size_t lm_id = 0;
        if (!LocateCellIdForPoint(p, col, page, lm_id)) return false;

        std::vector<Point>& vec = cell_list_[lm_id].local_data_;
        for (const auto& e : vec) {
            if (e == p) return false;
        }
        vec.push_back(p);
        return true;
    }

    size_t InsertBatch(const std::vector<Point>& pts) {
        if (pts.empty()) return 0;
        size_t ok = 0;

        struct Key { uint32_t col, page; size_t lm_id; };
        std::vector<std::pair<Key, const Point*> > bucket;
        bucket.reserve(pts.size());
        for (std::vector<Point>::const_iterator it = pts.begin(); it != pts.end(); ++it) {
            uint32_t c = 0, pg = 0; size_t lm = 0;
            if (!LocateCellIdForPoint(*it, c, pg, lm)) continue;
            Key k = { c, pg, lm };
            bucket.push_back(std::make_pair(k, &(*it)));
        }
        std::sort(bucket.begin(), bucket.end(),
            [](const std::pair<Key, const Point*>& a, const std::pair<Key, const Point*>& b) {
                return a.first.lm_id < b.first.lm_id;
            });

        size_t i = 0;
        while (i < bucket.size()) {
            size_t j = i;
            const size_t lm_id = bucket[i].first.lm_id;
            std::vector<Point>& vec = cell_list_[lm_id].local_data_;

            while (j < bucket.size() && bucket[j].first.lm_id == lm_id) {
                const Point& cand = *(bucket[j].second);
                bool exists = false;
                for (const auto& e : vec) { if (e == cand) { exists = true; break; } }
                if (!exists) { vec.push_back(cand); ++ok; }
                ++j;
            }
            i = j;
        }
        return ok;
    }

    bool Erase(const Point& p) {
        uint32_t col = 0, page = 0; size_t lm_id = 0;
        if (!LocateCellIdForPoint(p, col, page, lm_id)) return false;

        std::vector<Point>& vec = cell_list_[lm_id].local_data_;
        for (std::vector<Point>::iterator it = vec.begin(); it != vec.end(); ++it) {
            if (*it == p) { vec.erase(it); return true; }
        }
        return false;
    }

    size_t EraseBatch(const std::vector<Point>& pts) {
        if (pts.empty()) return 0;
        size_t ok = 0;

        struct Key { uint32_t col, page; size_t lm_id; };
        std::vector<std::pair<Key, const Point*> > bucket;
        bucket.reserve(pts.size());
        for (std::vector<Point>::const_iterator it = pts.begin(); it != pts.end(); ++it) {
            uint32_t c = 0, pg = 0; size_t lm = 0;
            if (!LocateCellIdForPoint(*it, c, pg, lm)) continue;
            Key k = { c, pg, lm };
            bucket.push_back(std::make_pair(k, &(*it)));
        }
        std::sort(bucket.begin(), bucket.end(),
            [](const std::pair<Key, const Point*>& a, const std::pair<Key, const Point*>& b) {
                return a.first.lm_id < b.first.lm_id;
            });

        size_t i = 0;
        while (i < bucket.size()) {
            size_t j = i;
            const size_t lm_id = bucket[i].first.lm_id;
            std::vector<Point>& vec = cell_list_[lm_id].local_data_;

            while (j < bucket.size() && bucket[j].first.lm_id == lm_id) {
                const Point& target = *(bucket[j].second);
                for (std::vector<Point>::iterator it = vec.begin(); it != vec.end(); ++it) {
                    if (*it == target) { vec.erase(it); ++ok; break; }
                }
                ++j;
            }
            i = j;
        }
        return ok;
    }


    size_t ModelSize() const {
        size_t sz = grid_split_.size() * sizeof(double);
        for (const auto& v : local_model_split_) sz += v.size() * sizeof(double);
        sz += sizeof(LocalModel) * page_cnt_;
        return sz;
    }

private:
    static bool PointWithinMBR(const Point& p, const Query& q) {
        for (int i = 0; i < Constants::DIM; ++i) {
            if (p.elements_[i] < q.low_.elements_[i]) return false;
            if (p.elements_[i] >= q.high_.elements_[i]) return false;
        }
        return true;
    }
    static Point InferQueryCenter(const Query& q) {
    #if QUERY_HAS_CENTER_RADIUS
            return q.center_;
    #else
            Point c{};
            for (int i = 0; i < Constants::DIM; ++i) c.elements_[i] = 0.5 * (q.low_.elements_[i] + q.high_.elements_[i]);
            return c;
    #endif
        }
    static bool PointWithinBall(const Point& p, const Point& center, double radius2) {
        double dist2 = 0.0;
        for (int i = 0; i < Constants::DIM; ++i) {
            double d = p.elements_[i] - center.elements_[i];
            dist2 += d * d;
            if (dist2 > radius2) return false;
        }
        return dist2 <= radius2;
    }

    inline std::pair<double, double> ColumnBounds(uint32_t col) const {
        double hi = grid_split_[col];
        double lo = (col == 0 ? -std::numeric_limits<double>::infinity()
            : grid_split_[col - 1]);
        return { lo, hi };
    }

    inline std::pair<double, double> PageBounds(uint32_t col, uint32_t page) const {
        const auto& split = local_model_split_[col];
        double hi = split[page];
        double lo = (page == 0 ? -std::numeric_limits<double>::infinity()
            : split[page - 1]);
        return { lo, hi };
    }

    std::pair<uint32_t, uint32_t> LocateCell(const Point& p) const {
        const int DIM = Constants::DIM;
        const int primary_axis = dim_order_[0];
        const int secondary_axis = (DIM >= 2 ? dim_order_[1] : dim_order_[0]);

        auto col_it = std::upper_bound(grid_split_.begin(), grid_split_.end(),
            p.elements_[primary_axis]);
        uint32_t col = uint32_t(col_it - grid_split_.begin());
        if (col >= num_grid_splits_) col = num_grid_splits_ - 1;

        const auto& page_split = local_model_split_[col];
        if (page_split.empty()) return { col, 0 };
        auto page_it = std::upper_bound(page_split.begin(), page_split.end(),
            p.elements_[secondary_axis]);
        uint32_t page = uint32_t(page_it - page_split.begin());
        if (page >= page_split.size()) page = uint32_t(page_split.size() - 1);

        return { col, page };
    }

    bool LocateCellIdForPoint(const Point& p, uint32_t& col, uint32_t& page, size_t& lm_id) const {
        if (grid_split_.empty() || local_model_split_.empty()) return false;

        const int DIM = Constants::DIM;
        const int primary_axis = dim_order_[0];
        const int secondary_axis = (DIM >= 2 ? dim_order_[1] : dim_order_[0]);

        std::vector<double>::const_iterator col_it =
            std::upper_bound(grid_split_.begin(), grid_split_.end(), p.elements_[primary_axis]);
        col = static_cast<uint32_t>(col_it - grid_split_.begin());
        if (col >= num_grid_splits_) col = num_grid_splits_ - 1;

        const std::vector<double>& page_split = local_model_split_[col];
        if (page_split.empty()) return false;
        std::vector<double>::const_iterator page_it =
            std::upper_bound(page_split.begin(), page_split.end(), p.elements_[secondary_axis]);
        page = static_cast<uint32_t>(page_it - page_split.begin());
        if (page >= page_split.size()) page = static_cast<uint32_t>(page_split.size() - 1);

        lm_id = local_model_ids_[col][page];
        return true;
    }
};




#endif