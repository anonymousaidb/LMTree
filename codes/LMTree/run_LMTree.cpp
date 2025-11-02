#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <print>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "ProgressBar.hpp"
#include "TimerClock.hpp"

#include "Parameters.hpp"

#include "indexes/NewLMTree.h"
#include "nlohmann/json.hpp"

std::int64_t count_equal_elements(const std::vector<std::int64_t> &arr1,
                                  const std::vector<std::int64_t> &arr2) {
    std::vector<std::int64_t> intersection;
    intersection.reserve(
        std::min(arr1.size(), arr2.size()));
    std::ranges::set_intersection(arr1, arr2, std::back_inserter(intersection));
    return static_cast<std::int64_t>(intersection.size());
}

std::int64_t check_result(std::vector<std::int64_t> result,
                          std::vector<std::int64_t> another_result) {
    std::ranges::sort(result);
    std::ranges::sort(another_result);
    return count_equal_elements(result, another_result);
}

std::vector<std::int64_t> range_ground_truth(const float *query,
                                             const float radius,
                                             const Dataset &dataset,
                                             std::int64_t end_id = -1) {
    std::vector<std::int64_t> result;
    if (end_id == -1) {
        end_id = dataset.Rows();
    }
    for (std::int64_t i = 0; i < end_id; i++) {
        auto data_ptr = dataset[i];
        if (const auto dist = distance(query, data_ptr); dist > radius) {
            continue;
        }
        result.push_back(i);
    }
    return result;
}

std::vector<std::int64_t> knn_ground_truth(const float *query, std::int64_t k,
                                           const Dataset &dataset) {
    std::priority_queue<std::pair<float, std::int64_t>> max_heap;

    for (std::int64_t i = 0; i < dataset.Rows(); ++i) {
        const float *data_ptr = dataset[i];
        const float dist = distance(query, data_ptr);

        if (max_heap.size() < k) {
            max_heap.emplace(dist, i);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i);
        }
    }

    std::vector<std::int64_t> result;
    result.reserve(k);
    while (!max_heap.empty()) {
        result.push_back(max_heap.top().second);
        max_heap.pop();
    }

    std::reverse(result.begin(), result.end());

    return result;
}

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::mutex result_mutex;

auto readDataFile(const std::string &filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("can not open file: " + filePath);
    }

    int cols = 0, rows = 0;
    std::string first_line;

    std::getline(file, first_line);
    std::istringstream iss(first_line);
    int extra_var_out;
    if (!(iss >> cols >> rows >> extra_var_out)) {
        throw std::runtime_error("parse failed 'cols rows extra_var'");
    }

    Dataset dataset(rows, cols);
    std::string line;
    for (int i = 0; i < rows && std::getline(file, line); ++i) {
        std::istringstream line_stream(line);
        auto *vector = dataset[i];
        for (int j = 0; j < cols; ++j) {
            if (!(line_stream >> vector[j])) {
                throw std::runtime_error("reading " + std::to_string(i + 1) +
                                         " line failed");
            }
        }
    }

    file.close();
    return dataset;
}

std::vector<float> co_selectivity = {0.558f, 0.622f, 0.768f, 0.812f, 0.962};
std::vector<float> la_selectivity = {989.f, 1409.f, 1875.f, 2314.f, 3090.f};
std::vector<float> sy_selectivity = {3229.f, 3843.f, 4614.f, 5613.f, 7090.f};

std::vector<std::int64_t> knn_list = {20, 50, 100, 150, 200};

std::unordered_map<std::string, std::unordered_map<std::string, nlohmann::json>>
    query_time_record;

void query(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else if (dataset_name == "blob") {
        prefix = "50-100-datasets/blob100D_";
    } else if (dataset_name == "mnist") {
        prefix = "50-100-datasets/mnist50D_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    std::vector<std::int64_t> size_list = {200000, 400000, 600000, 800000,
                                           1000000};
    nlohmann::json result;
    for (auto size : size_list) {
        auto dataset = readDataFile(father_path + "multi_dim/" + prefix +
                                    std::to_string(size) + ".txt");
        std::vector<float> selectivity_list;
        if (dataset_name == "CO") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 1;
            selectivity_list = co_selectivity;
        } else if (dataset_name == "LA") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 2;
            selectivity_list = la_selectivity;
        } else if (dataset_name == "SY") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 0;
            selectivity_list = sy_selectivity;
        } else {
            throw std::runtime_error("bad dataset name 1");
        }
        TimerClock tc;
        tc.tick();
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < dataset.Rows(); ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << mtree_time << ":" << modeling_time << ":" << merging_time
                  << std::endl;

        nlohmann::json sub_result;
        sub_result["building"] = build_tc.second();
        sub_result["memory"] = index.calculateNodeMemory(index.root);
        sub_result["nodes-count"] = index.calculate_nodes_count(index.root);
        cost = 0;
        for (std::int64_t selectivity_id = 0;
             selectivity_id < selectivity_list.size(); ++selectivity_id) {
            auto radius = selectivity_list[selectivity_id];
            tc.tick();
            for (std::int64_t i = 0; i < queries.Rows(); ++i) {
                auto query = queries[i];
                auto index_result =
                    index.range_query_with_pivot(query, radius, dataset);
                if (0) {
                    auto ground_truth =
                        range_ground_truth(query, radius, dataset);
                    auto checked_result =
                        check_result(index_result, ground_truth);
                    auto recall =
                        static_cast<float>(checked_result) /
                        std::max(ground_truth.size(), index_result.size());
                    std::stringstream buf;
                    buf << index_result.size() << ":" << ground_truth.size()
                        << " ";
                    buf << "recall:" << recall;
                    if (recall < 1) {
                        puts("recall < 1");
                        std::cout << buf.str() << std::endl;
                    }
                }
            }

            sub_result["range-query"][std::to_string(selectivity_id)]["time"] =
                tc.nanoSec() / queries.Rows();
            sub_result["range-query"][std::to_string(selectivity_id)]["cost"] =
                static_cast<double>(cost) / (dataset.Rows() * queries.Rows());
            std::cout << std::scientific;

            std::cout << std::setprecision(10);
            std::cout << std::endl
                      << "cost:"
                      << static_cast<double>(cost) /
                             (dataset.Rows() * queries.Rows())
                      << std::endl;
            std::cout << std::endl
                      << "node_cost:"
                      << static_cast<double>(node_cost) /
                             (dataset.Rows() * queries.Rows())
                      << std::endl;

            std::cout << std::defaultfloat;
            cost = 0;
        }
        for (std::int64_t knn_id = 0; knn_id < knn_list.size(); ++knn_id) {
            auto k = knn_list[knn_id];
            tc.tick();
            for (std::int64_t i = 0; i < queries.Rows(); ++i) {
                auto query = queries[i];
                auto index_result =
                    index.kNN_query_with_pivot(query, k, dataset);
                if (0) {
                    auto ground_truth = knn_ground_truth(query, k, dataset);
                    auto checked_result =
                        check_result(index_result, ground_truth);
                    auto recall =
                        static_cast<float>(checked_result) /
                        std::max(ground_truth.size(), index_result.size());
                    std::stringstream buf;
                    buf << index_result.size() << ":" << ground_truth.size()
                        << " ";
                    buf << "recall:" << recall;
                    if (recall < 1) {
                        puts("recall < 1");
                        std::cout << buf.str() << std::endl;
                    }
                }
            }

            sub_result["knn-query"][std::to_string(k)]["time"] =
                tc.nanoSec() / queries.Rows();
            sub_result["knn-query"][std::to_string(k)]["cost"] =
                static_cast<double>(cost) / queries.Rows();
            std::cout << std::scientific;

            std::cout << std::setprecision(10);
            std::cout << std::endl
                      << "cost:"
                      << static_cast<double>(cost) /
                             (dataset.Rows() * queries.Rows())
                      << std::endl;
            std::cout << std::endl
                      << "node_cost:"
                      << static_cast<double>(node_cost) /
                             (dataset.Rows() * queries.Rows())
                      << std::endl;

            std::cout << std::defaultfloat;
            cost = 0;
        }
        std::cout << sub_result << std::endl;
        result[std::to_string(size)] = sub_result;
    }
    std::cout << result << std::endl;
    std::ofstream result_file(dataset_name + "-fanout" +
                              std::to_string(DEFAULT_FANOUT) + "-epsilon" +
                              std::to_string(epsilon) + ".json");
    result_file << result.dump(6);
}

void query_var(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else if (dataset_name == "blob") {
        prefix = "50-100-datasets/blob100D_";
    } else if (dataset_name == "mnist") {
        prefix = "50-100-datasets/mnist50D_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    std::vector<std::int64_t> size_list = {200000, 400000, 600000, 800000,
                                           1000000};
    nlohmann::json result;
    for (auto size : size_list) {
        auto dataset = readDataFile(father_path + "multi_dim/" + prefix +
                                    std::to_string(size) + ".txt");
        std::vector<float> selectivity_list;
        if (dataset_name == "CO") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 1;
            selectivity_list = co_selectivity;
        } else if (dataset_name == "LA") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 2;
            selectivity_list = la_selectivity;
        } else if (dataset_name == "SY") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 0;
            selectivity_list = sy_selectivity;
        } else {
            throw std::runtime_error("bad dataset name 1");
        }
        TimerClock tc;
        tc.tick();
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < dataset.Rows(); ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << mtree_time << ":" << modeling_time << ":" << merging_time
                  << std::endl;

        nlohmann::json sub_result;
        sub_result["building"] = build_tc.second();
        sub_result["memory"] = index.calculateNodeMemory(index.root);
        sub_result["nodes-count"] = index.calculate_nodes_count(index.root);
        cost = 0;
        auto radius = selectivity_list[2];
        tc.tick();
        for (std::int64_t i = 0; i < queries.Rows(); ++i) {
            auto query = queries[i];
            auto index_result =
                index.range_query_with_pivot(query, radius, dataset);
            if (0) {
                auto ground_truth = range_ground_truth(query, radius, dataset);
                auto checked_result = check_result(index_result, ground_truth);
                auto recall =
                    static_cast<float>(checked_result) /
                    std::max(ground_truth.size(), index_result.size());
                std::stringstream buf;
                buf << index_result.size() << ":" << ground_truth.size() << " ";
                buf << "recall:" << recall;
                if (recall < 1) {
                    puts("recall < 1");
                    std::cout << buf.str() << std::endl;
                }
            }
        }
        {
            std::unique_lock lock(result_mutex);
            query_time_record[dataset_name][std::to_string(size)].push_back(
                tc.nanoSec() / queries.Rows());
        }
    }
}

void query_with_diff_dim(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else if (dataset_name == "blob") {
        prefix = "50-100-datasets/blob100D_";
    } else if (dataset_name == "mnist") {
        prefix = "50-100-datasets/mnist50D_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    auto dataset =
        readDataFile(father_path + "multi_dim/" + prefix + +"data.txt");
    std::vector<int> dims;
    for (int dim = 2; dim <= dataset.Cols(); dim *= 2) {
        dims.push_back(dim);
    }
    if (dims.back() != dataset.Cols()) {
        dims.push_back(dataset.Cols());
    }
    std::ranges::reverse(dims);
    std::vector<float> selectivity_list;
    if (dataset_name == "mnist") {
        selectivity_list = {4.15, 5.97, 7.54, 8.79, 9.64, 12.29};
    } else {
        selectivity_list = {4.35, 11.98, 21.94, 37.70, 60.47, 86.78, 113.36};
    }
    for (std::int64_t i = 0; i < dims.size(); i++) {
        node_cost = 0;
        cost = 0;
        int dim = dims[i];
        std::cout << dataset.Rows() << ":" << dataset.Cols() << std::endl;
        if (dim > dataset.Cols()) {
            dim = dataset.Cols();
        }
        dataset.cut_dim(dim);
        this_dim = dim;
        std::cout << "dataset.Cols():" << dataset.Cols() << std::endl;

        cost_o = this_dim;
        dist_type = 2;
        TimerClock tc;
        tc.tick();
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < dataset.Rows(); ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << mtree_time << ":" << modeling_time << ":" << merging_time
                  << std::endl;

        nlohmann::json sub_result;
        sub_result["building"] = build_tc.second();
        sub_result["memory"] = index.calculateNodeMemory(index.root);
        sub_result["nodes-count"] = index.calculate_nodes_count(index.root);
        cost = 0;

        auto radius = selectivity_list[i];
        tc.tick();
        for (std::int64_t i = 0; i < queries.Rows(); ++i) {
            auto query = dataset[i];
            auto index_result =
                index.range_query_with_pivot(query, radius, dataset);
            if (0) {
                auto ground_truth = range_ground_truth(query, radius, dataset);
                auto checked_result = check_result(index_result, ground_truth);
                auto recall =
                    static_cast<float>(checked_result) /
                    std::max(ground_truth.size(), index_result.size());
                std::stringstream buf;
                buf << index_result.size() << ":" << ground_truth.size() << " ";
                buf << "recall:" << recall;
                if (recall < 1) {
                    puts("recall < 1");
                    std::cout << buf.str() << std::endl;
                }
            }
        }

        sub_result["range-query"]["time"] = tc.nanoSec() / queries.Rows();
        sub_result["range-query"]["cost"] =
            static_cast<double>(cost) / (dataset.Rows() * queries.Rows());
        std::cout << std::scientific;

        std::cout << std::setprecision(10);
        std::cout << std::endl
                  << "cost:"
                  << static_cast<double>(cost) /
                         (dataset.Rows() * queries.Rows())
                  << std::endl;
        std::cout << std::endl
                  << "node_cost:"
                  << static_cast<double>(node_cost) /
                         (dataset.Rows() * queries.Rows())
                  << std::endl;

        std::cout << std::defaultfloat;
        cost = 0;
        int k = 100;
        for (std::int64_t i = 0; i < queries.Rows(); ++i) {
            auto query = dataset[i];
            auto index_result = index.kNN_query_with_pivot(query, k, dataset);
            if (0) {
                auto ground_truth = knn_ground_truth(query, k, dataset);
                auto checked_result = check_result(index_result, ground_truth);
                auto recall =
                    static_cast<float>(checked_result) /
                    std::max(ground_truth.size(), index_result.size());
                std::stringstream buf;
                buf << index_result.size() << ":" << ground_truth.size() << " ";
                buf << "recall:" << recall;
                if (recall < 1) {
                    puts("recall < 1");
                    std::cout << buf.str() << std::endl;
                }
            }
        }

        sub_result["knn-query"][std::to_string(k)]["time"] =
            tc.nanoSec() / queries.Rows();
        sub_result["knn-query"][std::to_string(k)]["cost"] =
            static_cast<double>(cost) / queries.Rows();
        std::cout << std::scientific;

        std::cout << std::setprecision(10);
        std::cout << std::endl
                  << "cost:"
                  << static_cast<double>(cost) /
                         (dataset.Rows() * queries.Rows())
                  << std::endl;
        std::cout << std::endl
                  << "node_cost:"
                  << static_cast<double>(node_cost) /
                         (dataset.Rows() * queries.Rows())
                  << std::endl;

        std::cout << std::defaultfloat;
        cost = 0;
        query_time_record[dataset_name][std::to_string(dim)] = sub_result;
    }
}

void diff_epsilon_query(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    nlohmann::json result;
    auto dataset = readDataFile(father_path + "multi_dim/" + prefix +
                                std::to_string(1000000) + ".txt");
    std::vector<float> selectivity_list;
    if (dataset_name == "CO") {
        this_dim = dataset.Cols();
        cost_o = this_dim;
        dist_type = 1;
        selectivity_list = co_selectivity;
    } else if (dataset_name == "LA") {
        this_dim = dataset.Cols();
        cost_o = this_dim;
        dist_type = 2;
        selectivity_list = la_selectivity;
    } else if (dataset_name == "SY") {
        this_dim = dataset.Cols();
        cost_o = this_dim;
        dist_type = 0;
        selectivity_list = sy_selectivity;
    } else {
        throw std::runtime_error("bad dataset name 1");
    }
    TimerClock tc;
    tc.tick();
    for (auto e : std::vector({1, 2, 4, 6, 8, 10, 16, 22, 28, 32})) {
        epsilon = e;
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < dataset.Rows(); ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << mtree_time << ":" << modeling_time << ":" << merging_time
                  << std::endl;

        nlohmann::json sub_result;
        sub_result["building"] = build_tc.second();
        sub_result["memory"] = index.calculateNodeMemory(index.root);
        sub_result["nodes-count"] = index.calculate_nodes_count(index.root);
        cost = 0;
        auto radius = selectivity_list[2];
        tc.tick();
        for (std::int64_t i = 0; i < queries.Rows(); ++i) {
            auto query = dataset[i];
            auto index_result =
                index.range_query_with_pivot(query, radius, dataset);
            if (0) {
                auto ground_truth = range_ground_truth(query, radius, dataset);
                auto checked_result = check_result(index_result, ground_truth);
                auto recall =
                    static_cast<float>(checked_result) /
                    std::max(ground_truth.size(), index_result.size());
                std::stringstream buf;
                buf << index_result.size() << ":" << ground_truth.size() << " ";
                buf << "recall:" << recall;
                if (recall < 1) {
                    puts("recall < 1");
                    std::cout << buf.str() << std::endl;
                }
            }
        }

        sub_result["time"] = tc.nanoSec() / queries.Rows();
        sub_result["cost"] = static_cast<double>(cost) / queries.Rows();
        cost = 0;
        std::cout << sub_result << std::endl;
        result[epsilon] = sub_result;
    }
    std::ofstream result_file(dataset_name + "-fanout" +
                              std::to_string(DEFAULT_FANOUT) + "-epsilon" +
                              std::to_string(epsilon) + "-diff-epsilon.json");
    result_file << result.dump(6);
}

void ablation_study_query(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    nlohmann::json result;
    auto dataset = readDataFile(father_path + "multi_dim/" + prefix +
                                std::to_string(1000000) + ".txt");
    std::vector<float> selectivity_list;
    if (dataset_name == "CO") {
        this_dim = dataset.Cols();
        cost_o = this_dim;
        dist_type = 1;
        selectivity_list = co_selectivity;
    } else if (dataset_name == "LA") {
        this_dim = dataset.Cols();
        cost_o = this_dim;
        dist_type = 2;
        selectivity_list = la_selectivity;
    } else if (dataset_name == "SY") {
        this_dim = dataset.Cols();
        cost_o = this_dim;
        dist_type = 0;
        selectivity_list = sy_selectivity;
    } else {
        throw std::runtime_error("bad dataset name 1");
    }
    TimerClock tc;
    tc.tick();
    {
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < dataset.Rows(); ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << mtree_time << ":" << modeling_time << ":" << merging_time
                  << std::endl;

        nlohmann::json sub_result;
        sub_result["building"] = build_tc.second();
        sub_result["memory"] = index.calculateNodeMemory(index.root);
        sub_result["nodes-count"] = index.calculate_nodes_count(index.root);
        cost = 0;
        auto radius = selectivity_list[2];
        tc.tick();
        for (std::int64_t i = 0; i < queries.Rows(); ++i) {
            auto query = dataset[i];
            auto index_result =
                index.range_query_with_pivot(query, radius, dataset);
            if (0) {
                auto ground_truth = range_ground_truth(query, radius, dataset);
                auto checked_result = check_result(index_result, ground_truth);
                auto recall =
                    static_cast<float>(checked_result) /
                    std::max(ground_truth.size(), index_result.size());
                std::stringstream buf;
                buf << index_result.size() << ":" << ground_truth.size() << " ";
                buf << "recall:" << recall;
                if (recall < 1) {
                    puts("recall < 1");
                    std::cout << buf.str() << std::endl;
                }
            }
        }

        sub_result["time"] = tc.nanoSec() / queries.Rows();
        sub_result["cost"] = static_cast<double>(cost) / queries.Rows();
        cost = 0;
        std::cout << sub_result << std::endl;
        result["with-merge"] = sub_result;
    }
    {
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < dataset.Rows(); ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        auto merging_time = tc.second();
        std::cout << mtree_time << ":" << modeling_time << ":" << merging_time
                  << std::endl;

        nlohmann::json sub_result;
        sub_result["building"] = build_tc.second();
        sub_result["memory"] = index.calculateNodeMemory(index.root);
        sub_result["nodes-count"] = index.calculate_nodes_count(index.root);
        cost = 0;
        auto radius = selectivity_list[2];
        tc.tick();
        for (std::int64_t i = 0; i < queries.Rows(); ++i) {
            auto query = dataset[i];
            auto index_result =
                index.range_query_with_pivot(query, radius, dataset);
            if (0) {
                auto ground_truth = range_ground_truth(query, radius, dataset);
                auto checked_result = check_result(index_result, ground_truth);
                auto recall =
                    static_cast<float>(checked_result) /
                    std::max(ground_truth.size(), index_result.size());
                std::stringstream buf;
                buf << index_result.size() << ":" << ground_truth.size() << " ";
                buf << "recall:" << recall;
                if (recall < 1) {
                    puts("recall < 1");
                    std::cout << buf.str() << std::endl;
                }
            }
        }

        sub_result["time"] = tc.nanoSec() / queries.Rows();
        sub_result["cost"] = static_cast<double>(cost) / queries.Rows();
        cost = 0;
        std::cout << sub_result << std::endl;
        result["without-merge"] = sub_result;
    }
    std::ofstream result_file(dataset_name + "-fanout" +
                              std::to_string(DEFAULT_FANOUT) + "-epsilon" +
                              std::to_string(epsilon) + "-ablation_study.json");
    result_file << result.dump(6);
}

auto merge_dataset(Dataset &sub_set1, Dataset &sub_set2) {
    Dataset result(sub_set1.Rows() + sub_set2.Rows(), sub_set1.Cols());
    for (std::int64_t i = 0; i < sub_set1.Rows(); ++i) {
        std::memcpy(result[i], sub_set1[i], sub_set1.Cols() * sizeof(float));
    }
    for (std::int64_t i = 0; i < sub_set2.Rows(); ++i) {
        std::memcpy(result[i + sub_set1.Rows()], sub_set2[i],
                    sub_set2.Cols() * sizeof(float));
    }
    return result;
}

std::unordered_map<std::string,
                   std::unordered_map<std::string, std::vector<double>>>
    insert_time_record;
std::unordered_map<std::string,
                   std::unordered_map<std::string, std::vector<double>>>
    delete_time_record;

void insert_per_batch(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto to_insert_data = readDataFile(father_path + "multi_dim/" + prefix +
                                       std::to_string(400000) + ".txt");
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    std::vector<std::int64_t> size_list = {600000};
    nlohmann::json result;
    for (auto size : size_list) {
        auto dataset_ = readDataFile(father_path + "multi_dim/" + prefix +
                                     std::to_string(size) + ".txt");
        auto dataset = merge_dataset(dataset_, to_insert_data);
        std::vector<float> selectivity_list;
        if (dataset_name == "CO") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 1;
            selectivity_list = co_selectivity;
        } else if (dataset_name == "LA") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 2;
            selectivity_list = la_selectivity;
        } else if (dataset_name == "SY") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 0;
            selectivity_list = sy_selectivity;
        } else {
            throw std::runtime_error("bad dataset name 1");
        }
        TimerClock tc;
        tc.tick();
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < size; ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << "Thread ID: " << pthread_self() << std::endl;
        std::cout << size << ":" << mtree_time << ":" << modeling_time << ":"
                  << merging_time << std::endl;

        cost = 0;
        auto selectivity_id = 2;
        std::int64_t insert_time_acc = 0;
        std::int64_t delete_time_acc = 0;
        std::int64_t query_time_acc = 0;
        std::int64_t query_cost_acc = 0;
        ProgressBar bar;
        for (std::int64_t insert_batch = 0; insert_batch < 8; ++insert_batch) {
            cost = node_cost = 0;
            std::int64_t insert_start_id = size + insert_batch * 50 * 1000;
            std::int64_t insert_stop_id = size + (insert_batch + 1) * 50 * 1000;
            tc.tick();
            for (std::int64_t insert_id = insert_start_id;
                 insert_id < insert_stop_id; ++insert_id) {
                index.insert_with_pivot(insert_id, dataset);
                if (insert_id % 1000 == 0) {
                    bar.update(static_cast<double>(insert_id % (50 * 1000)) /
                                   (50 * 1000),
                               "");
                }
            }
            insert_time_acc = tc.nanoSec();
            auto radius = selectivity_list[selectivity_id];
            tc.tick();

            cost = node_cost = 0;
            for (std::int64_t i = 0; i < queries.Rows(); ++i) {
                break;
                auto query = dataset[i];
                auto index_result =
                    index.range_query_with_pivot(query, radius, dataset);
                if (0) {
                    auto ground_truth = range_ground_truth(
                        query, radius, dataset, insert_stop_id);
                    auto checked_result =
                        check_result(index_result, ground_truth);
                    auto recall =
                        static_cast<float>(checked_result) /
                        std::max(ground_truth.size(), index_result.size());
                    std::stringstream buf;
                    buf << index_result.size() << ":" << ground_truth.size()
                        << " ";
                    buf << "recall:" << recall;
                    if (recall < 1) {
                        puts("recall < 1");
                        std::cout << buf.str() << std::endl;
                    }
                }
            }
            query_time_acc += tc.nanoSec();
            query_cost_acc += cost;
            {
                std::unique_lock lock(result_mutex);
                insert_time_record[dataset_name]["total"].push_back(
                    double(insert_time_acc) / (50 * 1000));
                insert_time_record[dataset_name]["split"].push_back(
                    double(split_time) / (50 * 1000));
                insert_time_record[dataset_name]["merge"].push_back(
                    double(merging_time) / (50 * 1000));
                merging_time = 0;
                split_time = 0;
                insert_time_acc = 0;
            }
        }

        for (std::int64_t delete_batch = 7; delete_batch >= 0; --delete_batch) {
            std::int64_t delete_start_id = size + delete_batch * 50 * 1000;
            std::int64_t delete_stop_id = size + (delete_batch + 1) * 50 * 1000;

            tc.tick();
            for (std::int64_t delete_id = delete_start_id;
                 delete_id < delete_stop_id; ++delete_id) {
                index.delete_with_pivot(delete_id, dataset);
            }
            delete_time_acc = tc.nanoSec();
            auto radius = selectivity_list[selectivity_id];
            tc.tick();
            cost = node_cost = 0;
            for (std::int64_t i = 0; i < queries.Rows(); ++i) {
                auto query = dataset[i];
                auto index_result =
                    index.range_query_with_pivot(query, radius, dataset);
                if (0) {
                    auto ground_truth = range_ground_truth(
                        query, radius, dataset, delete_start_id);
                    auto checked_result =
                        check_result(index_result, ground_truth);
                    auto recall =
                        static_cast<float>(checked_result) /
                        std::max(ground_truth.size(), index_result.size());
                    std::stringstream buf;
                    buf << index_result.size() << ":" << ground_truth.size()
                        << " ";
                    buf << "recall:" << recall;
                    if (recall < 1) {
                        puts("recall < 1");
                        std::cout << buf.str() << std::endl;
                    }
                }
            }
            query_cost_acc += cost;
            query_time_acc += tc.nanoSec();
            {
                std::unique_lock lock(result_mutex);
                delete_time_record[dataset_name]["total"].push_back(
                    double(delete_time_acc) / (50 * 1000));
                delete_time_record[dataset_name]["split"].push_back(
                    double(split_time) / (50 * 1000));
                delete_time_record[dataset_name]["merge"].push_back(
                    double(merging_time) / (50 * 1000));
                merging_time = 0;
                split_time = 0;
                insert_time_acc = 0;
            }
        }
    }
    std::ofstream result_file(dataset_name + "-fanout" +
                              std::to_string(DEFAULT_FANOUT) + "-epsilon" +
                              std::to_string(epsilon) + "-insert.json");
    result_file << result.dump(6);
}

void insert_per_100(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto to_insert_data = readDataFile(father_path + "multi_dim/" + prefix +
                                       std::to_string(400000) + ".txt");
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    std::vector<std::int64_t> size_list = {600000};
    nlohmann::json result;
    for (auto size : size_list) {
        auto dataset_ = readDataFile(father_path + "multi_dim/" + prefix +
                                     std::to_string(size) + ".txt");
        auto dataset = merge_dataset(dataset_, to_insert_data);
        std::vector<float> selectivity_list;
        if (dataset_name == "CO") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 1;
            selectivity_list = co_selectivity;
        } else if (dataset_name == "LA") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 2;
            selectivity_list = la_selectivity;
        } else if (dataset_name == "SY") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 0;
            selectivity_list = sy_selectivity;
        } else {
            throw std::runtime_error("bad dataset name 1");
        }
        TimerClock tc;
        tc.tick();
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < size; ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << "Thread ID: " << pthread_self() << std::endl;
        std::cout << size << ":" << mtree_time << ":" << modeling_time << ":"
                  << merging_time << std::endl;

        cost = 0;
        std::int64_t insert_time_acc = 0;
        std::int64_t delete_time_acc = 0;
        std::int64_t query_time_acc = 0;
        std::int64_t query_cost_acc = 0;
        ProgressBar bar;
        tc.tick();
        for (std::int64_t insert_id = 0; insert_id < (400 * 1000);
             ++insert_id) {
            index.insert_with_pivot(insert_id, dataset);
            if (insert_id % 100 == 0) {
                bar.update(static_cast<double>(insert_id % (400 * 1000)) /
                               (400 * 1000),
                           "");
                insert_time_acc = tc.nanoSec();
                {
                    std::unique_lock lock(result_mutex);
                    insert_time_record[dataset_name]["total"].push_back(
                        double(insert_time_acc) / (100));
                }
                tc.tick();
            }
        }
    }
}

void insert_delete_per_100(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto to_insert_data = readDataFile(father_path + "multi_dim/" + prefix +
                                       std::to_string(400000) + ".txt");
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    std::vector<std::int64_t> size_list = {600000};
    nlohmann::json result;
    for (auto size : size_list) {
        auto dataset_ = readDataFile(father_path + "multi_dim/" + prefix +
                                     std::to_string(size) + ".txt");
        auto dataset = merge_dataset(dataset_, to_insert_data);
        std::vector<float> selectivity_list;
        if (dataset_name == "CO") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 1;
            selectivity_list = co_selectivity;
        } else if (dataset_name == "LA") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 2;
            selectivity_list = la_selectivity;
        } else if (dataset_name == "SY") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 0;
            selectivity_list = sy_selectivity;
        } else {
            throw std::runtime_error("bad dataset name 1");
        }
        TimerClock tc;
        tc.tick();
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < size; ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << "Thread ID: " << pthread_self() << std::endl;
        std::cout << size << ":" << mtree_time << ":" << modeling_time << ":"
                  << merging_time << std::endl;

        cost = 0;
        std::int64_t insert_time_acc = 0;
        std::int64_t delete_time_acc = 0;
        std::int64_t query_time_acc = 0;
        std::int64_t query_cost_acc = 0;
        ProgressBar bar;
        tc.tick();
        const std::int64_t N = 400 * 1000;

        std::vector<double> local_total;
        local_total.reserve(N / 100 + 8);


        const std::int64_t BATCH_SIZE = 100;
        local_total.reserve(N / 100 + 8);
        merge_times = 0;
        split_times = 0;
        re_segment_times = 0;
        for (std::int64_t batch_begin = 0; batch_begin < N; batch_begin += BATCH_SIZE) {
            const std::int64_t batch_end = std::min(batch_begin + BATCH_SIZE, N);

            for (std::int64_t insert_id = batch_begin; insert_id < batch_end; ++insert_id) {
                index.insert_with_pivot(insert_id + size, dataset);
            }
            for (std::int64_t insert_id = batch_begin; insert_id < batch_end; ++insert_id) {
                index.delete_with_pivot(insert_id, dataset);
            }
             bar.update(static_cast<double>(batch_begin) / N, "");
            auto insert_time_acc = tc.nanoSec();
            local_total.push_back(static_cast<double>(insert_time_acc) / 100.0);

            tc.tick();
        }

        {
            std::unique_lock lock(result_mutex);
            auto& vec = insert_time_record[dataset_name]["total"];
            vec.insert(vec.end(), local_total.begin(), local_total.end());
        }
        std::cout << "split_times:"<<split_times << std::endl;
        std::cout << "merge_times:"<<merge_times << std::endl;
        std::cout << "re_segment_times:"<<re_segment_times << std::endl;
    }
}

void delete_per_100(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto to_insert_data = readDataFile(father_path + "multi_dim/" + prefix +
                                       std::to_string(400000) + ".txt");
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    std::vector<std::int64_t> size_list = {600000};
    nlohmann::json result;
    for (auto size : size_list) {
        auto dataset_ = readDataFile(father_path + "multi_dim/" + prefix +
                                     std::to_string(size) + ".txt");
        auto dataset = merge_dataset(dataset_, to_insert_data);
        std::vector<float> selectivity_list;
        if (dataset_name == "CO") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 1;
            selectivity_list = co_selectivity;
        } else if (dataset_name == "LA") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 2;
            selectivity_list = la_selectivity;
        } else if (dataset_name == "SY") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 0;
            selectivity_list = sy_selectivity;
        } else {
            throw std::runtime_error("bad dataset name 1");
        }
        TimerClock tc;
        tc.tick();
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < size; ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << "Thread ID: " << pthread_self() << std::endl;
        std::cout << size << ":" << mtree_time << ":" << modeling_time << ":"
                  << merging_time << std::endl;

        cost = 0;
        std::int64_t insert_time_acc = 0;
        std::int64_t delete_time_acc = 0;
        std::int64_t query_time_acc = 0;
        std::int64_t query_cost_acc = 0;
        ProgressBar bar;
        tc.tick();
        const std::int64_t N = 400 * 1000;

        std::vector<double> local_total;
        local_total.reserve(N / 100 + 8);

        for (std::int64_t insert_id = 0; insert_id < N; ++insert_id) {
            index.delete_with_pivot(insert_id, dataset);

            if (insert_id % 100 == 0) {
                bar.update(static_cast<double>(insert_id) / N, "");

                auto insert_time_acc = tc.nanoSec();
                local_total.push_back(static_cast<double>(insert_time_acc) /   100.0);

                tc.tick();
            }
        }

        {
            std::unique_lock lock(result_mutex);
            auto &vec = insert_time_record[dataset_name]["total"];
            vec.insert(vec.end(), local_total.begin(), local_total.end());
        }
    }
}

void insert(std::string dataset_name) {
    std::string prefix = "";
    if (dataset_name == "CO") {
        prefix = "CO/color_32_";
    } else if (dataset_name == "LA") {
        prefix = "LA/LA_";
    } else if (dataset_name == "SY") {
        prefix = "SY/SY_";
    } else {
        throw std::runtime_error("bad dataset name 0");
    }
    auto to_insert_data = readDataFile(father_path + "multi_dim/" + prefix +
                                       std::to_string(400000) + ".txt");
    auto queries =
        readDataFile(father_path + "multi_dim/" + prefix + "query.txt");
    std::vector<std::int64_t> size_list = {200000, 400000, 600000, 800000,
                                           1000000};
    nlohmann::json result;
    for (auto size : size_list) {
        auto dataset_ = readDataFile(father_path + "multi_dim/" + prefix +
                                     std::to_string(size) + ".txt");
        auto dataset = merge_dataset(dataset_, to_insert_data);
        std::vector<float> selectivity_list;
        if (dataset_name == "CO") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 1;
            selectivity_list = co_selectivity;
        } else if (dataset_name == "LA") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 2;
            selectivity_list = la_selectivity;
        } else if (dataset_name == "SY") {
            this_dim = dataset.Cols();
            cost_o = this_dim;
            dist_type = 0;
            selectivity_list = sy_selectivity;
        } else {
            throw std::runtime_error("bad dataset name 1");
        }
        TimerClock tc;
        tc.tick();
        NewLMTree index;
        TimerClock build_tc;
        for (std::int64_t i = 0; i < size; ++i) {
            index.insert(i, dataset);
        }
        auto mtree_time = tc.second();
        tc.tick();
        index.add_models(dataset);
        auto modeling_time = tc.second();
        tc.tick();
        index.merge(dataset);
        auto merging_time = tc.second();
        std::cout << "Thread ID: " << pthread_self() << std::endl;
        std::cout << size << ":" << mtree_time << ":" << modeling_time << ":"
                  << merging_time << std::endl;

        nlohmann::json sub_result;
        sub_result["building"] = build_tc.second();
        sub_result["memory"] = index.calculateNodeMemory(index.root);
        sub_result["nodes-count"] = index.calculate_nodes_count(index.root);
        cost = 0;
        auto selectivity_id = 2;
        std::int64_t insert_time_acc = 0;
        std::int64_t delete_time_acc = 0;
        std::int64_t query_time_acc = 0;
        std::int64_t query_cost_acc = 0;
        ProgressBar bar;
        for (std::int64_t insert_batch = 0; insert_batch < 5; ++insert_batch) {
            cost = node_cost = 0;
            std::int64_t insert_start_id = size + insert_batch * 80 * 1000;
            std::int64_t insert_stop_id = size + (insert_batch + 1) * 80 * 1000;
            tc.tick();
            for (std::int64_t insert_id = insert_start_id;
                 insert_id < insert_stop_id; ++insert_id) {
                index.insert_with_pivot(insert_id, dataset);
                if (insert_id % 1000 == 0) {
                    bar.update(static_cast<double>(insert_id % (80 * 1000)) /
                                   (80 * 1000),
                               "");
                }
            }
            insert_time_acc += tc.nanoSec();
            auto radius = selectivity_list[selectivity_id];
            tc.tick();

            cost = node_cost = 0;
            for (std::int64_t i = 0; i < queries.Rows(); ++i) {

                auto query = dataset[i];
                auto index_result =
                    index.range_query_with_pivot(query, radius, dataset);
                if (0) {
                    auto ground_truth = range_ground_truth(
                        query, radius, dataset, insert_stop_id);
                    auto checked_result =
                        check_result(index_result, ground_truth);
                    auto recall =
                        static_cast<float>(checked_result) /
                        std::max(ground_truth.size(), index_result.size());
                    std::stringstream buf;
                    buf << index_result.size() << ":" << ground_truth.size()
                        << " ";
                    buf << "recall:" << recall;
                    if (recall < 1) {
                        puts("recall < 1");
                        std::cout << buf.str() << std::endl;
                    }
                }
            }
            query_time_acc += tc.nanoSec();
            query_cost_acc += cost;
        }
        {
            std::unique_lock<std::mutex> lock(result_mutex);
            insert_time_record[dataset_name][std::to_string(size)].push_back(
                insert_time_acc / (5 * 80 * 1000));
            sub_result["insert-time"] = insert_time_acc / (5 * 80 * 1000);
            result[std::to_string(size)] = sub_result;
        }

        std::cout << sub_result << std::endl;
        for (std::int64_t delete_batch = 4; delete_batch >= 0; --delete_batch) {
            std::int64_t delete_start_id = size + delete_batch * 80 * 1000;
            std::int64_t delete_stop_id = size + (delete_batch + 1) * 80 * 1000;

            tc.tick();
            for (std::int64_t delete_id = delete_start_id;
                 delete_id < delete_stop_id; ++delete_id) {
                index.delete_with_pivot(delete_id, dataset);
            }
            delete_time_acc += tc.nanoSec();
            auto radius = selectivity_list[selectivity_id];
            tc.tick();
            cost = node_cost = 0;
            for (std::int64_t i = 0; i < queries.Rows(); ++i) {
                auto query = dataset[i];
                auto index_result =
                    index.range_query_with_pivot(query, radius, dataset);
                if (0) {
                    auto ground_truth = range_ground_truth(
                        query, radius, dataset, delete_start_id);
                    auto checked_result =
                        check_result(index_result, ground_truth);
                    auto recall =
                        static_cast<float>(checked_result) /
                        std::max(ground_truth.size(), index_result.size());
                    std::stringstream buf;
                    buf << index_result.size() << ":" << ground_truth.size()
                        << " ";
                    buf << "recall:" << recall;
                    if (recall < 1) {
                        puts("recall < 1");
                        std::cout << buf.str() << std::endl;
                    }
                }
            }
            query_cost_acc += cost;
            query_time_acc += tc.nanoSec();
        }
        sub_result["delete-time"] = delete_time_acc / (5 * 80 * 1000);
        sub_result["query-time"] = query_time_acc / (2 * 5 * queries.Rows());
        sub_result["query-cost"] = query_cost_acc / (2 * 5 * queries.Rows());
        std::cout << sub_result << std::endl;
        result[std::to_string(size)] = sub_result;
    }
    std::ofstream result_file(dataset_name + "-fanout" +
                              std::to_string(DEFAULT_FANOUT) + "-epsilon" +
                              std::to_string(epsilon) + "-insert.json");
    result_file << result.dump(6);
}

template <typename Func> void run_in_child_process(Func f, const char *arg) {
    pid_t pid = fork();
    if (pid == 0) {
        f(arg);
        _exit(0);
    } else if (pid > 0) {
    } else {
        perror("fork");
    }
}

void wait_child_processes() {
    while (waitpid(-1, nullptr, 0) > 0)
        ;
}

void parallel_test() {
    DEFAULT_FANOUT = 1024;
    epsilon = 5;
    run_in_child_process(ablation_study_query, "CO");
    run_in_child_process(ablation_study_query, "LA");
    run_in_child_process(ablation_study_query, "SY");
    wait_child_processes();
    run_in_child_process(diff_epsilon_query, "CO");
    run_in_child_process(diff_epsilon_query, "LA");
    run_in_child_process(diff_epsilon_query, "SY");

    wait_child_processes();
    run_in_child_process(query, "CO");
    run_in_child_process(query, "LA");
    run_in_child_process(query, "SY");
    wait_child_processes();

    run_in_child_process(insert, "CO");
    run_in_child_process(insert, "LA");
    run_in_child_process(insert, "SY");

    wait_child_processes();
    std::cout << "All children finished.\n";
}

#include <execinfo.h>
#include <signal.h>

void handler(int) {
    void *array[20];
    int size = backtrace(array, 20);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
}

void test_insert() {
    std::vector<std::thread> workers;
    workers.reserve(5);
    for (int i = 0; i < 5; ++i) {
        workers.emplace_back([&] { insert("SY"); });
    }
    for (auto &t : workers)
        t.join();

    workers.clear();
    for (int i = 0; i < 5; ++i) {
        workers.emplace_back([&] { insert("CO"); });
    }
    for (auto &t : workers)
        t.join();

    nlohmann::json j;
    for (auto &[k1, inner_map] : insert_time_record) {
        for (auto &[k2, vec] : inner_map) {
            j[k1][k2] = vec;
        }
    }
    std::println("{}", j.dump());
}


int main1() {

    DEFAULT_FANOUT = 1024;
    epsilon = 5;


    std::vector<std::string> codes = {"LA", "CO", "SY"};
    for (const auto &code : codes) {
        insert_per_100(code);
    }

    nlohmann::json j;
    for (auto &[k1, inner_map] : insert_time_record) {
        for (auto &[k2, vec] : inner_map) {
            j[k1][k2] = vec;
        }
    }
    std::ofstream output_file1(
        "output-insert.json");
    if (output_file1.is_open()) {
        output_file1 << j.dump(6) << std::endl;
        output_file1.close();
    } else {
        std::cerr << "canout open" << std::endl;
    }
    insert_time_record.clear();
    for (const auto &code : codes) {
        delete_per_100(code);
    }

    for (auto &[k1, inner_map] : insert_time_record) {
        for (auto &[k2, vec] : inner_map) {
            j[k1][k2] = vec;
        }
    }

    std::ofstream output_file(
        "output-delete.json");
    if (output_file.is_open()) {
        output_file << j.dump(6) << std::endl;
        output_file.close();
    } else {
        std::cerr << "canout open" << std::endl;
    }

    return 0;
}

int main_insert_delete() {
    DEFAULT_FANOUT = 1024;
    epsilon = 5;
    insert_delete_per_100("CO");

    nlohmann::json j;
    for (auto &[k1, inner_map] : insert_time_record) {
        for (auto &[k2, vec] : inner_map) {
            j[k1][k2] = vec;
        }
    }
    std::ofstream output_file(
        "output-insert-delete.json");
    if (output_file.is_open()) {
        output_file << j.dump(6) << std::endl;
        output_file.close();
    } else {
        std::cerr << "canout open" << std::endl;
    }
    return 0;
}

int main2() {
    DEFAULT_FANOUT = 1024;
    epsilon = 5;
    insert_per_batch("CO");

    std::vector<std::thread> threads;
    for (auto &thread : threads) {
        thread.join();
    }

    nlohmann::json j;
    for (auto &[k1, inner_map] : insert_time_record) {
        for (auto &[k2, vec] : inner_map) {
            j[k1][k2] = vec;
        }
    }
    std::println("{}", j.dump(6));
    for (auto &[k1, inner_map] : delete_time_record) {
        for (auto &[k2, vec] : inner_map) {
            j[k1][k2] = vec;
        }
    }
    std::println("{}", j.dump(6));

    return 0;
}

int main_query() {

    std::vector<std::thread> workers;
    workers.reserve(5);
    for (auto dataset : std::vector<std::string>({"CO", "LA", "SY"})) {
        workers.clear();
        for (int i = 0; i < 5; ++i) {
            workers.emplace_back([&] { query_var(dataset); });
        }
        for (auto &t : workers)
            t.join();
    }
    nlohmann::json j;

    for (auto &[outer_key, inner_map] : query_time_record) {
        for (auto &[inner_key, value] : inner_map) {
            j[outer_key][inner_key] = value;
        }
    }

    std::cout << j.dump(6) << std::endl;
    return 0;
}

void insert_exp() {
    insert("LA");
    insert("CO");
}

int main() {

    main_insert_delete();
    return 0;
}