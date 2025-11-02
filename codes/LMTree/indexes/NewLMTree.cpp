#include "NewLMTree.h"

#include "../TimerClock.hpp"
#include <fstream>
#include <functional>
#include <queue>
#include <ranges>

#include "IndexParameters.h"
#include "basic.h"


bool NewLMTree::compare_node_vectors_unsorted(const std::vector<Node *> &a, const std::vector<Node *> &b) {
    if (a.size() != b.size()) return false;

    std::vector<Node *> sorted_a = a;
    std::vector<Node *> sorted_b = b;

    std::ranges::sort(sorted_a);
    std::ranges::sort(sorted_b);

    return std::equal(sorted_a.begin(), sorted_a.end(), sorted_b.begin());
}

bool NewLMTree::compare_data_vectors_unsorted(const std::vector<std::int64_t> &a, const std::vector<std::int64_t> &b) {
    if (a.size() != b.size()) return false;

    std::vector<std::int64_t> sorted_a = a;
    std::vector<std::int64_t> sorted_b = b;

    std::ranges::sort(sorted_a);
    std::ranges::sort(sorted_b);

    return std::equal(sorted_a.begin(), sorted_a.end(), sorted_b.begin());
}

void NewLMTree::Node::replace_child(Node *child, Node *new_child, float new_distance) {
    for (auto &seg: segments) {
        for (auto &item: seg.items) {
            if (item.second.child == child) {
                item.second.child = new_child;
                item.first = new_distance;
                return;
            }
        }
    }
}

void NewLMTree::Node::remove_child(Node *child) {
    bool deleted = false;
    for (auto seg_it = segments.begin(); seg_it != segments.end(); ++seg_it) {
        for (auto it = seg_it->items.begin(); it != seg_it->items.end(); ++it) {
            if (it->second.child == child) {
                it = seg_it->items.erase(it);
                deleted = true;
                break;
            }
        }
        if (seg_it->items.empty()) {
            segments.erase(seg_it);
        }
        if (deleted) {
            return;
        }
    }
}

std::int64_t NewLMTree::Node::size() const {
    std::int64_t size = 0;
    for (auto &seg: segments) {
        size += seg.items.size();
    }
    return size;
}

std::int64_t NewLMTree::Node::leaf_node_count() const {
    if (is_leaf) {
        return 0;
    }

    std::int64_t size = 0;
    for (auto &seg: segments) {
        for (auto &item: seg.items) {
            if (item.second.child->is_leaf) {
                ++size;
            }
        }
    }
    return size;
}

std::vector<NewLMTree::Node *> NewLMTree::Node::leaf_children() {
    std::vector<Node *> result;
    if (is_leaf) {
        throw std::runtime_error("Leaf node type error");
    }

    std::int64_t size = 0;
    for (auto &seg: segments) {
        for (auto &item: seg.items) {
            if (item.second.child->is_leaf) {
                result.emplace_back(item.second.child);
            }
        }
    }
    return result;
}

std::vector<NewLMTree::Node *> NewLMTree::Node::all_children() const {
    std::vector<Node *> result;
    for (auto &seg: segments) {
        for (auto &item: seg.items) {
            result.push_back(item.second.child);
        }
    }
    return result;
}

void NewLMTree::Node::check_adding_pivot() const {
    if (!is_leaf) {
        std::vector<Node *> old_nodes;
        std::vector<Node *> new_nodes;
        for (auto &item: items) {
            old_nodes.push_back(item.second.child);
        }
        for (auto &seg: segments) {
            for (auto &item: seg.items) {
                new_nodes.push_back(item.second.child);
            }
        }
        if (old_nodes.size() != new_nodes.size() || !compare_node_vectors_unsorted(old_nodes, new_nodes)) {
            std::cout << old_nodes.size() << ":" << new_nodes.size() << std::endl;
            throw std::runtime_error("Nodes not matched");
        }
    } else {
        std::vector<std::int64_t> old_nodes;
        std::vector<std::int64_t> new_nodes;
        for (auto &item: items) {
            old_nodes.push_back(item.second.id);
        }
        for (auto &seg: segments) {
            for (auto &item: seg.items) {
                new_nodes.push_back(item.second.id);
            }
        }
        if (old_nodes.size() != new_nodes.size() || !compare_data_vectors_unsorted(old_nodes, new_nodes)) {
            std::cout << old_nodes.size() << ":" << new_nodes.size() << std::endl;
            throw std::runtime_error("Nodes not matched");
        }
    }
}

void NewLMTree::Node::add_data(float dist_to_center, const std::int64_t id) {
    items.emplace_back(dist_to_center, Item(id));
    radius = std::max(radius, dist_to_center);
}

void NewLMTree::Node::add_sub_node(float dist_to_center, Node *node) {
    items.emplace_back(dist_to_center, Item(node));
    radius = std::max(radius, dist_to_center + node->radius);
}

void NewLMTree::Node::set_sub_node(const std::int64_t index, float dist_to_center, Node *node) {
    items[index] = {dist_to_center, Item(node)};
    radius = 0;
    for (const auto &[dist_to_center, child]: items) {
        radius = std::max(radius, dist_to_center + child.child->radius);
    }
}


bool NewLMTree::Node::is_full() const noexcept {
    return items.size() >= DEFAULT_FANOUT;
}

void NewLMTree::print_tree(const Node *node, int depth) {
    std::cout << std::string(depth * 2, ' ');
    std::cout << "Node: " << node->center;
    if (node->is_leaf && !node->items.empty()) {
        std::cout << " [Data: ";
        for (const auto &entry: node->items) {
            std::cout << entry.second.id << " ";
        }
        std::cout << "]";
    }
    std::cout << std::endl;
    for (const auto &child: node->items) {
        print_tree(child.second.child, depth + 1);
    }
}


void NewLMTree::try_merge(Node *node, Node *father, Dataset &dataset) {
    ++merge_times;
    TimerClock tc;
    const std::function<void(Node &, Node &, Node &)> merge_two_node
            = [&dataset](Node &new_node, Node &ei, Node &ej) {
        new_node.center = ei.center;
        new_node.radius = 0;
        for (auto &seg: ei.segments) {
            for (auto &item: seg.items) {
                new_node.add_data(item.first, item.second.id);
            }
        }
        for (auto &seg: ej.segments) {
            for (auto &item: seg.items) {
                auto dist = distance(dataset[new_node.center], dataset[item.second.id]);
                new_node.add_data(dist, item.second.id);
            }
        }
        add_pivot_model_for_leaf(new_node, dataset);
    };

    while (true) {
        auto children = node->leaf_children();
        if (children.size() <= 1 || children.size() > 100) {
            break;
        }
        auto delta_costs = std::vector(children.size(), std::vector<float>(children.size()));
        auto prepare_nodes = std::vector(children.size(), std::vector<Node *>(children.size()));


        for (std::int64_t i = 0; i < children.size(); ++i) {

            for (std::int64_t j = 0; j < children.size(); ++j) {

                auto ei = children[i];
                auto ej = children[j];
                if (i == j) {
                    delta_costs[i][j] = 0;
                } else if (ei->segments.size() > 1 || ej->segments.size() > 1) {
                    delta_costs[i][j] = -1;
                } else {
                    if (0) {
                        const float min_dist_from_ej_to_ei = std::max(static_cast<float>(0),
                                                                      distance(dataset[ei->center],
                                                                               dataset[ej->center]) -
                                                                      ej->radius);
                        auto &seg = ei->segments.front();
                        auto predicted_position_in_ei = static_cast<std::int64_t>(seg.forward(min_dist_from_ej_to_ei));
                        assert(std::isfinite(predicted_position_in_ei));

                        auto real_position = std::lower_bound(
                                                                           seg.items.begin(), seg.items.end(),
                                                                           min_dist_from_ej_to_ei,
                                                                           [](const auto &pair, float value) {
                                                                               return pair.first < value;
                                                                           }
                                                                       ) - seg.items.begin();
                        if (std::abs(predicted_position_in_ei - real_position) > epsilon) {
                            delta_costs[i][j] = -1;
                            continue;
                        }
                    }

                    auto &new_node = prepare_nodes[i][j];
                    new_node = new Node();
                    new_node->is_leaf = true;
                    merge_two_node(*new_node, *ei, *ej);
                    delta_costs[i][j] = ei->cost + ej->cost - new_node->cost;
                }
            }
        }

        std::int64_t max_i = -1, max_j = -1;
        float max_delta_cost = -std::numeric_limits<float>::max();
        for (std::int64_t i = 0; i < delta_costs.size(); ++i) {
            for (std::int64_t j = 0; j < delta_costs[i].size(); ++j) {
                if (delta_costs[i][j] > max_delta_cost) {
                    max_delta_cost = delta_costs[i][j];
                    max_i = i;
                    max_j = j;
                }
            }
        }
        if (max_delta_cost <= 0) {
            break;
        }
        assert(max_i != -1 && max_j != -1);
        assert(max_i != max_j);
        Node *new_node = prepare_nodes[max_i][max_j];
        node->replace_child(children[max_i], new_node,
                            distance(dataset[children[max_i]->center], dataset[new_node->center]));
        node->remove_child(children[max_j]);
        delete children[max_i];
        delete children[max_j];
        for (std::int64_t i = 0; i < prepare_nodes.size(); ++i) {
            for (std::int64_t j = 0; j < prepare_nodes[i].size(); ++j) {
                if (i == max_i && j == max_j) {
                    continue;
                }
                if (prepare_nodes[i][j])
                delete prepare_nodes[i][j];
            }
        }
    }

    node->update_size();
    if (node->size() == 1) {
        auto child = node->all_children().front();
        if (node == root) {
            delete node;
            root = child;
        } else {
            auto dist = distance(dataset[father->center], dataset[child->center]);
            father->replace_child(node, child, dist);
            delete node;
        }
    }
    merge_time += tc.nanoSec();
}

void NewLMTree::merge(Node *node, Node *father, Dataset &dataset) {
    if (node->is_leaf) {
        return;
    }
    for (auto &seg: node->segments) {
        for (auto &item: seg.items) {
            merge(item.second.child, node, dataset);
        }
    }
    try_merge(node, father, dataset);
}

NewLMTree::~NewLMTree() {
    std::queue<Node *> routes;
    routes.push(root);
    while (!routes.empty()) {
        auto node = routes.front();
        routes.pop();
        if (!node->is_leaf) {
            for (auto &seg: node->segments) {
                for (auto item: seg.items) {
                    routes.push(item.second.child);
                }
            }
        }
        delete node;
    }
}

void NewLMTree::check_adding_pivot() const {
    std::queue<const Node *> routes;
    routes.push(root);
    while (!routes.empty()) {
        auto node = routes.front();
        routes.pop();
        node->check_adding_pivot();
        if (!node->is_leaf) {
            for (auto &item: node->items) {
                routes.push(item.second.child);
            }
        }
    }
}

void NewLMTree::add_pivot_model_for_leaf(Node &node, Dataset &dataset) {
    node.segments.clear();
    std::ranges::sort(node.items, [](auto &a, auto &b) {
        return a.first < b.first;
    });
    std::vector<std::pair<float, float> > to_split_tmp_data;
    float max_dist = -std::numeric_limits<float>::max();
    float min_dist = std::numeric_limits<float>::max();
    for (std::int64_t i = 0; i < node.items.size(); i++) {
        auto &[fst, snd] = node.items[i];
        to_split_tmp_data.emplace_back(fst, static_cast<float>(i));
        min_dist = std::min(min_dist, fst);
        max_dist = std::max(max_dist, fst);
    }
    const auto mean_slope = static_cast<float>(node.items.size() - 1) / (max_dist - min_dist);
    for (const auto linear_models = ShrinkingCone_Segmentation(to_split_tmp_data, epsilon);
         const auto [begin_idx,end_idx,slope,intercept]: linear_models) {
        Segment seg;
        seg.min_dist_to_center = node.items[begin_idx].first;
        seg.max_dist_to_center = node.items[end_idx].first;
        if (seg.slope > mean_slope) {
            seg.pivot = node.items[begin_idx].second.id;
            for (std::int64_t j = begin_idx; j <= end_idx; j++) {
                auto &item = node.items[j];
                seg.items.emplace_back(distance(dataset[seg.pivot], dataset[item.second.id]), item.second.id);
            }
            std::ranges::sort(seg.items,
                              [](auto &a, auto &b) {
                                  return a.first < b.first;
                              });
        } else {
            seg.pivot = node.center;
            for (std::int64_t j = begin_idx; j <= end_idx; j++) {
                auto &item = node.items[j];
                seg.items.emplace_back(item);
            }
        }
        node.segments.emplace_back(std::move(seg));
    }
    for (auto &seg: node.segments) {
        if (seg.items.size() == 1) {
            seg.slope = 0;
            seg.intercept = 0;
            seg.error_bound = 0;
            continue;
        }
        std::vector<std::pair<float, float> > tmp_data;
        for (std::int64_t i = 0; i < seg.items.size(); i++) {
            tmp_data.emplace_back(seg.items[i].first, static_cast<float>(i));
        }
        auto [slope,intercept,max_error] = linearFit(tmp_data);
        seg.slope = slope;
        seg.intercept = intercept;
        seg.error_bound = max_error;
    }

    float error_accumulated = 0;
    for (auto &i: node.segments) {
        error_accumulated += i.error_bound;
    }
    node.cost = node.segments.size() * cost_o + error_accumulated;
    node.last_split_cost = node.cost;
}

void NewLMTree::add_pivot_model_for_inner(Node &node, Dataset &dataset) {
    node.segments.clear();
    std::ranges::sort(node.items, [](auto &a, auto &b) {
        return a.first - a.second.child->radius < b.first - b.second.child->radius;
    });
    std::vector<std::pair<float, float> > to_split_tmp_data;
    float min_min_dist = std::numeric_limits<float>::max();
    float max_min_dist = -std::numeric_limits<float>::max();
    for (std::int64_t i = 0; i < node.items.size(); i++) {
        auto &[dist_fc_cc, child] = node.items[i];
        auto key = dist_fc_cc - child.child->radius;
        to_split_tmp_data.emplace_back(key, static_cast<float>(i));
        min_min_dist = std::min(min_min_dist, key);
        max_min_dist = std::max(max_min_dist, key);
    }
    const auto mean_slope = static_cast<float>(node.items.size() - 1) / (max_min_dist - min_min_dist);
    const auto linear_models = ShrinkingCone_Segmentation(to_split_tmp_data, epsilon);
    for (const auto [begin_idx, end_idx, slope, intercept]: linear_models) {
        Segment seg;
        seg.min_dist_to_center = std::numeric_limits<float>::max();
        seg.max_dist_to_center = -std::numeric_limits<float>::max();
        if (seg.slope > mean_slope) {
            seg.pivot = node.items[begin_idx].second.child->center;
        } else {
            seg.pivot = node.center;
        }
        for (auto it = node.items.begin() + begin_idx, end_it = node.items.begin() + end_idx + 1; it < end_it; it++) {
            auto min_dist_to_center = it->first - it->second.child->radius;
            auto max_dist_to_center = it->first + it->second.child->radius;
            seg.min_dist_to_center = std::min(seg.min_dist_to_center, min_dist_to_center);
            seg.max_dist_to_center = std::max(seg.max_dist_to_center, max_dist_to_center);
            if (seg.pivot != node.center) {
                seg.items.emplace_back(distance(dataset[seg.pivot], dataset[it->second.child->center]), it->second);
            } else {
                seg.items.emplace_back(*it);
            }
        }

        if (seg.pivot != node.center) {
            std::ranges::sort(seg.items, [](auto &a, auto &b) {
                return a.first - a.second.child->radius < b.first - b.second.child->radius;
            });
        }
        node.segments.emplace_back(std::move(seg));
    }
    for (auto &seg: node.segments) {
        if (seg.items.size() == 1) {
            seg.slope = 0;
            seg.intercept = 0;
            seg.error_bound = 0;
            continue;
        }
        std::vector<std::pair<float, float> > tmp_data;
        for (std::int64_t i = 0; i < seg.items.size(); i++) {
            auto &item = seg.items[i];
            tmp_data.emplace_back(item.first - item.second.child->radius, static_cast<float>(i));
        }
        auto [slope,intercept,max_error] = linearFit(tmp_data);
        seg.slope = slope;
        seg.intercept = intercept;
        seg.error_bound = max_error;
    }
    float error_accumulated = 0;
    for (const auto &i: node.segments) {
        error_accumulated += i.error_bound;
    }
    node.old_seg_size = node.segments.size();
    node.cost = static_cast<float>(node.segments.size()) * cost_o + error_accumulated;
    node.last_split_cost = node.cost;
}

void NewLMTree::add_models(Dataset &dataset) {
    std::queue<Node *> queue;
    queue.push(root);
    while (!queue.empty()) {
        auto node = queue.front();
        node->update_size();
        queue.pop();
        if (node->is_leaf) {
            add_pivot_model_for_leaf(*node, dataset);
        } else {
            for (auto &val: node->items | std::views::values) {
                queue.push(val.child);
            }
            add_pivot_model_for_inner(*node, dataset);
        }
        node->check_adding_pivot();
    }
}


void NewLMTree::insert(std::int64_t id, Dataset &dataset) {
    if (root->is_leaf && root->items.empty()) {
        root->center = id;
        root->add_data(distance(dataset[id], dataset[root->center]), id);
        return;
    }
    insert(id, dataset, root, nullptr, -1);
}

template <typename Iterator>
Iterator it_min(Iterator it1, Iterator it2) {
    return (*it1 < *it2) ? it1 : it2;
}

template <typename Iterator>
Iterator it_max(Iterator it1, Iterator it2) {
    return (*it1 > *it2) ? it1 : it2;
}

bool NewLMTree::insert_with_pivot(const std::int64_t insert_id, Dataset &dataset, Node *node, Segment *father_seg,
                                  Node *father_node, const std::int64_t index_in_father) {
    const auto dist_to_center = distance(dataset[insert_id], dataset[node->center]);
    node->radius = std::max(node->radius, dist_to_center);
    if (node->is_leaf) {
        std::int64_t nearest_seg_id = -1;
        float nearest_seg_dist = std::numeric_limits<float>::max();
        assert(!node->segments.empty());
        if (dist_to_center < node->segments.front().min_dist_to_center) {
            nearest_seg_id = 0;
        } else if (dist_to_center > node->segments.back().max_dist_to_center) {
            nearest_seg_id = static_cast<std::int64_t>(node->segments.size()) - 1;
        } else {
            for (std::int64_t i = 0; i < node->segments.size(); i++) {
                const auto &seg = node->segments[i];
                const auto dist = std::max(seg.min_dist_to_center - dist_to_center,
                                           dist_to_center - seg.max_dist_to_center);
                if (dist < nearest_seg_dist) {
                    nearest_seg_dist = dist;
                    nearest_seg_id = i;
                    break;
                }
            }
        }
        assert(nearest_seg_id != -1);
        auto &seg = node->segments[nearest_seg_id];
        seg.min_dist_to_center = std::min(seg.min_dist_to_center, dist_to_center);
        seg.max_dist_to_center = std::max(seg.max_dist_to_center, dist_to_center);

        auto dist_to_pivot = distance(dataset[insert_id], dataset[seg.pivot]);

        auto pre_position = static_cast<std::ptrdiff_t>(seg.forward(dist_to_pivot));

        auto start_offset = std::max<std::int64_t>(0,
            std::min<std::int64_t>(seg.items.size(),
                pre_position - static_cast<std::int64_t>(seg.error_bound + 1)));
        auto end_offset = std::max<std::int64_t>(0,
         std::min<std::int64_t>(seg.items.size(),
             pre_position + static_cast<std::int64_t>(seg.error_bound + 1)));
        auto lo = seg.items.begin() + start_offset;
        auto hi = seg.items.begin() + end_offset;

        auto it = std::lower_bound(lo, hi, dist_to_pivot,
            [](const auto& p, float v){ return p.first < v; });

        if (it == hi && hi != seg.items.end()) {
            it = std::lower_bound(hi, seg.items.end(), dist_to_pivot,
                [](const auto& p, float v){ return p.first < v; });
        } else if (it == lo && lo != seg.items.begin()) {
            it = std::lower_bound(seg.items.begin(), lo, dist_to_pivot,
                [](const auto& p, float v){ return p.first < v; });
        }
        if (it < seg.items.begin() || it > seg.items.end()) {
            std::cerr << "size=" << seg.items.size()
          << " start=" << start_offset
          << " end=" << end_offset
          << " pre_pos=" << pre_position
          << std::endl;

        }

        seg.items.insert(it, {dist_to_pivot, Item(insert_id)});
        ++inserted_count;
        ++seg.error_bound;
        ++node->cost;
    } else {
        std::int64_t nearest_seg_id = -1;
        float nearest_seg_dist = std::numeric_limits<float>::max();
        if (dist_to_center < node->segments.front().min_dist_to_center) {
            nearest_seg_id = 0;
        } else if (dist_to_center > node->segments.back().max_dist_to_center) {
            nearest_seg_id = static_cast<std::int64_t>(node->segments.size()) - 1;
        } else {
            for (std::int64_t i = 0; i < node->segments.size(); i++) {
                const auto &seg = node->segments[i];
                if (const auto dist =
                            std::max(seg.min_dist_to_center - dist_to_center,
                                     dist_to_center - seg.max_dist_to_center);
                    dist < nearest_seg_dist) {
                    nearest_seg_dist = dist;
                    nearest_seg_id = i;
                    break;
                }
            }
        }
        assert(nearest_seg_id != -1);
        auto &seg = node->segments[nearest_seg_id];
        seg.min_dist_to_center = std::min(seg.min_dist_to_center, dist_to_center);
        seg.max_dist_to_center = std::max(seg.max_dist_to_center, dist_to_center);
        std::int64_t closest_child_id = 0;
        float min_dist = std::numeric_limits<float>::max();
        for (std::int64_t i = 0; i < seg.items.size(); ++i) {
            const auto child = seg.items[i].second.child;
            if (const auto dist = distance(dataset[child->center], dataset[insert_id]);
                dist < min_dist) {
                min_dist = dist;
                closest_child_id = i;
            }
        }
        const auto changed = insert_with_pivot(insert_id, dataset, seg.items[closest_child_id].second.child, &seg, node,
                                               closest_child_id);
        if (changed && node->size() > 2 * node->old_children_size) {
            try_merge(node, father_node, dataset);
        }
    }
    if (node->cost > 2 * node->last_split_cost) {
        if (node->segments.size() > 2 * node->old_children_size && node->size() > 2) {
            ++split_times;
            TimerClock timer;
            const auto new_node1 = new Node();
            const auto new_node2 = new Node();
            if (node->is_leaf) {
                new_node1->center = node->segments.front().items.front().second.id;
                new_node1->is_leaf = new_node2->is_leaf = true;
                float max_dist = -1;
                for (const auto &seg: node->segments) {
                    for (auto &item: seg.items) {
                        const auto to_split_data_id = item.second.id;
                        if (const auto dist = distance(dataset[new_node1->center], dataset[to_split_data_id]);
                            dist > max_dist) {
                            max_dist = dist;
                            new_node2->center = to_split_data_id;
                        }
                    }
                }
                for (auto &seg: node->segments) {
                    for (auto &item: seg.items) {
                        const auto to_split_data_id = item.second.id;
                        auto d1 = distance(dataset[new_node1->center], dataset[to_split_data_id]);
                        auto d2 = distance(dataset[new_node2->center], dataset[to_split_data_id]);
                        if (d1 == d2) {
                            if (new_node1->items.size() < new_node2->items.size()) {
                                new_node1->add_data(d1, to_split_data_id);
                            } else {
                                new_node2->add_data(d2, to_split_data_id);
                            }
                        } else {
                            if (d1 < d2) {
                                new_node1->add_data(d1, to_split_data_id);
                            } else {
                                new_node2->add_data(d2, to_split_data_id);
                            }
                        }
                    }
                }
                add_pivot_model_for_leaf(*new_node1, dataset);
                add_pivot_model_for_leaf(*new_node2, dataset);
                if(new_node1->segments.empty()){
                    throw std::runtime_error("Empty segment after split");
                }
                if(new_node2->segments.empty()){
                    throw std::runtime_error("Empty segment after split");
                }
            } else {
                new_node1->is_leaf = new_node2->is_leaf = false;
                new_node1->center = node->segments.front().items.front().second.child->center;
                float max_dist = -1;
                for (const auto &seg: node->segments) {
                    for (auto &item: seg.items) {
                        const auto to_split_data_id = item.second.child->center;
                        if (const auto dist = distance(dataset[new_node1->center], dataset[to_split_data_id]);
                            dist > max_dist) {
                            max_dist = dist;
                            new_node2->center = to_split_data_id;
                        }
                    }
                }
                for (auto &seg: node->segments) {
                    for (auto &item: seg.items) {
                        const auto to_split_data_id = item.second.child->center;
                        auto d1 = distance(dataset[new_node1->center], dataset[to_split_data_id]);
                        auto d2 = distance(dataset[new_node2->center], dataset[to_split_data_id]);
                        if (d1 == d2) {
                            if (new_node1->items.size() < new_node2->items.size()) {
                                new_node1->add_sub_node(d1, item.second.child);
                            } else {
                                new_node2->add_sub_node(d2, item.second.child);
                            }
                        } else {
                            if (d1 < d2) {
                                new_node1->add_sub_node(d1, item.second.child);
                            } else {
                                new_node2->add_sub_node(d2, item.second.child);
                            }
                        }
                    }
                }
                add_pivot_model_for_inner(*new_node1, dataset);
                add_pivot_model_for_inner(*new_node2, dataset);
            }
            if (node == root) {
                root = new Node();
                root->is_leaf = false;
                root->center = new_node1->center;
                auto d1 = distance(dataset[root->center], dataset[new_node1->center]);
                auto d2 = distance(dataset[root->center], dataset[new_node2->center]);
                root->add_sub_node(d1, new_node1);
                root->add_sub_node(d2, new_node2);
                add_pivot_model_for_inner(*root, dataset);
            } else {
                auto d1 = distance(dataset[father_seg->pivot], dataset[new_node1->center]);
                auto d2 = distance(dataset[father_seg->pivot], dataset[new_node2->center]);
                auto old_cost = father_node->cost;
                father_seg->items[index_in_father].second.child = new_node1;
                father_seg->items[index_in_father].first = d1;
                father_seg->min_dist_to_center = std::min(father_seg->min_dist_to_center,
                                                          father_seg->items[index_in_father].first - new_node1->radius);
                father_seg->max_dist_to_center = std::max(father_seg->max_dist_to_center,
                                                          father_seg->items[index_in_father].first + new_node1->radius);
                father_seg->items.emplace_back(d2, Item(new_node2));

                father_seg->min_dist_to_center = std::min(father_seg->min_dist_to_center,
                                                          father_seg->items.back().first - new_node2->radius);
                father_seg->max_dist_to_center = std::max(father_seg->max_dist_to_center,
                                                          father_seg->items.back().first + new_node2->radius);

                father_seg->error_bound++;
                father_node->cost++;
                father_node->last_split_cost = old_cost;
            }
            split_time += timer.nanoSec();
            delete node;
            return true;
        }
    }
    return false;
}

void NewLMTree::delete_with_pivot(const std::int64_t delete_id, Dataset &dataset, Node *node, Segment *father_seg,
                                  Node *father_node, const std::int64_t index_in_father) {
    const auto dist_to_center = distance(dataset[delete_id], dataset[node->center]);
    node->radius = std::max(node->radius, dist_to_center);
    if (node->is_leaf) {
        std::int64_t nearest_seg_id = -1;
        for (std::int64_t i = 0; i < node->segments.size(); i++) {
            auto &seg = node->segments[i];
            if (dist_to_center < seg.min_dist_to_center || dist_to_center > seg.max_dist_to_center) {
                continue;
            }
            auto dist_to_pivot = distance(dataset[delete_id], dataset[seg.pivot]);

            auto it = std::lower_bound(
                seg.items.begin(),
                seg.items.end(),
                dist_to_pivot,
                [](const std::pair<float, Item> &pair, const float value) {
                    return pair.first < value;
                }
            );

            while (it < seg.items.end() && it->first == dist_to_pivot) {
                if (it->second.id == delete_id) {
                    it = seg.items.erase(it);
                    break;
                }
                ++it;
            }
            ++seg.error_bound;
            ++node->cost;
        }
    } else {
        for (auto &seg: node->segments) {
            if (dist_to_center < seg.min_dist_to_center || dist_to_center > seg.max_dist_to_center) {
                continue;
            }
            for (auto it = seg.items.begin(); it < seg.items.end(); ++it) {
                const auto child = it->second.child;
                if (const auto dist = distance(dataset[child->center], dataset[delete_id]);
                    dist <= child->radius) {
                    delete_with_pivot(delete_id, dataset, child, &seg, node,
                                      it - seg.items.begin());
                }
            }
        }
    }
    if (node->cost > 10 * node->last_split_cost) {
        ++re_segment_times;
        node->items.clear();
        for (auto &seg: node->segments) {
            node->items.insert(node->items.end(), seg.items.begin(), seg.items.end());
        }
        if (node->is_leaf) {
            add_pivot_model_for_leaf(*node, dataset);
        } else {
            add_pivot_model_for_inner(*node, dataset);
        }
    }
}

std::vector<std::int64_t> NewLMTree::
kNN_query_with_pivot(const float *query, std::int64_t k, const Dataset &dataset) const {
    class kNNElement {
    public:
        const Node *node;
        const Segment *seg;
        bool is_seg = false;
        float dist_to_pivot = -1;
        float dist_to_center = -1;
    };
    auto cmp_node = [&](const std::pair<float, kNNElement> &first, const std::pair<float, kNNElement> &second) {
        return first.first > second.first;
    };
    std::priority_queue<std::pair<float, kNNElement>, std::vector<std::pair<float, kNNElement> >, decltype(cmp_node)>
            nodePQ(cmp_node);

    auto cmp_data = [&](const std::pair<float, std::int64_t> &first,
                        const std::pair<float, std::int64_t> &second) {
        return first.first < second.first;
    };
    std::priority_queue<std::pair<float, std::int64_t>,
        std::vector<std::pair<float, std::int64_t> >, decltype(cmp_data)> dataPQ(cmp_data); {
        kNNElement root_element;
        root_element.node = root;
        root_element.is_seg = false;
        root_element.dist_to_center = distance(dataset[root_element.node->center], query);
        nodePQ.emplace(root_element.dist_to_center - root_element.node->radius, root_element);
    }
    while (!nodePQ.empty()) {
        const kNNElement element = nodePQ.top().second;
        nodePQ.pop();
        if (element.node->is_leaf) {
            if (element.is_seg) {
                auto &seg = *element.seg;
                for (auto &item: seg.items) {
                    auto pivot_predict_dist = element.dist_to_pivot - item.first;
                    if (dataPQ.size() >= k && pivot_predict_dist > dataPQ.top().first) {
                        continue;
                    }
                    auto dist = distance(dataset[item.second.id], query);
                    cost++;
                    dataPQ.emplace(dist, item.second.id);
                    if (dataPQ.size() > k) {
                        dataPQ.pop();
                    }
                }
            } else {
                for (std::int64_t i = 0; i < element.node->segments.size(); ++i) {
                    auto new_element = element;
                    new_element.is_seg = true;
                    auto &seg = new_element.node->segments[i];
                    auto predicted_dist = std::max(seg.min_dist_to_center - new_element.dist_to_center,
                                                   new_element.dist_to_center - seg.max_dist_to_center);
                    if (dataPQ.size() >= k && predicted_dist > dataPQ.top().first) {
                        continue;
                    }
                    new_element.dist_to_pivot = distance(dataset[seg.pivot], query);
                    new_element.seg = &seg;
                    nodePQ.emplace(predicted_dist, new_element);
                }
            }
        } else {
            if (element.is_seg) {
                auto &seg = *element.seg;
                for (auto item: seg.items) {
                    const auto pivot_predict_dist = element.dist_to_pivot - item.first - item.second.child->radius;
                    if (dataPQ.size() >= k && pivot_predict_dist > dataPQ.top().first) {
                        continue;
                    }
                    auto dist = distance(dataset[item.second.child->center], query);
                    kNNElement new_element;
                    new_element.is_seg = false;
                    new_element.node = item.second.child;
                    new_element.dist_to_center = dist;
                    nodePQ.emplace(dist - item.second.child->radius, new_element);
                }
            } else {
                for (auto &seg: element.node->segments) {
                    auto new_element = element;
                    auto predicted_dist = std::max(seg.min_dist_to_center - element.dist_to_center,
                                                   element.dist_to_center - seg.max_dist_to_center);
                    if (dataPQ.size() >= k && predicted_dist > dataPQ.top().first) {
                        continue;
                    }
                    new_element.seg = &seg;
                    new_element.dist_to_pivot = distance(dataset[seg.pivot], query);
                    new_element.is_seg = true;
                    nodePQ.emplace(predicted_dist, new_element);
                }
            }
        }
        if (dataPQ.size() >= k && dataPQ.top().first < nodePQ.top().first) {
            break;
        }
    }
    std::vector<std::int64_t> result;
    while (!dataPQ.empty()) {
        result.push_back(dataPQ.top().second);
        dataPQ.pop();
    }
    return result;
}

std::vector<std::int64_t> NewLMTree::range_query_with_pivot(const float *query, const float radius,
                                                            const Dataset &dataset) const {
    std::vector<std::int64_t> result;
    std::queue<const Node *> routes;
    routes.push(root);
    while (!routes.empty()) {
        ++node_cost;
        const auto node = routes.front();
        routes.pop();
        const auto dist_query_center = distance(query, dataset[node->center]);
        if (dist_query_center > radius + node->radius) {
            continue;
        }
        if (node->is_leaf) {
            const auto min_dist_to_center = dist_query_center - radius;
            const auto max_dist_to_center = dist_query_center + radius;
            for (const auto &seg: node->segments) {
                if (seg.max_dist_to_center < min_dist_to_center || seg.min_dist_to_center > max_dist_to_center) {
                    continue;
                }

                const float dist_query_pivot = (seg.pivot != node->center)
                                                   ? distance(query, dataset[seg.pivot])
                                                   : dist_query_center;

                const auto min_dist = dist_query_pivot - radius;
                const auto max_dist = dist_query_pivot + radius;

                auto pre_position = static_cast<std::ptrdiff_t>(seg.forward(min_dist));

                auto start_offset = std::max<std::int64_t>(0,
                    std::min<std::int64_t>(seg.items.size(),
                        pre_position - static_cast<std::int64_t>(seg.error_bound + 1)));
                auto end_offset = std::max<std::int64_t>(0,
                 std::min<std::int64_t>(seg.items.size(),
                     pre_position + static_cast<std::int64_t>(seg.error_bound + 1)));
                const auto begin_idx = std::lower_bound(
                    seg.items.begin() + start_offset, seg.items.begin() + end_offset,
                    min_dist,
                    [](const auto &pair, float value) {
                        return pair.first < value;
                    }
                ) - seg.items.begin();


                auto end_idx =seg.items.size();


                for (std::int64_t i = begin_idx; i < end_idx; i++) {
                    const auto dist_to_pivot = seg.items[i].first;
                    const auto id = seg.items[i].second.id;
                    if (dist_to_pivot < min_dist) {
                        continue;
                    }
                    if (dist_to_pivot > max_dist) {
                        break;
                    }
                    const auto dist = distance(query, dataset[id]);

                    ++cost;
                    if (dist > radius) {
                        continue;
                    }
                    result.emplace_back(id);
                }
            }
        } else {
            const auto min_dist_to_center = dist_query_center - radius;
            const auto max_dist_to_center = dist_query_center + radius;
            for (const auto &seg: node->segments) {
                if (seg.max_dist_to_center < min_dist_to_center || seg.min_dist_to_center > max_dist_to_center) {
                    continue;
                }
                const float dist_query_pivot = (seg.pivot != node->center)
                                                   ? distance(query, dataset[seg.pivot])
                                                   : dist_query_center;

                auto min_dist = dist_query_pivot - radius;
                auto max_dist = dist_query_pivot + radius;
                auto begin_idx = static_cast<std::int64_t>(0);
                auto end_idx = seg.items.size();
                for (std::int64_t i = begin_idx; i < end_idx; i++) {
                    const auto dist_to_pivot = seg.items[i].first;
                    const auto child = seg.items[i].second.child;
                    if (dist_to_pivot + child->radius < min_dist || dist_to_pivot - child->radius > max_dist) {
                        continue;
                    }
                    routes.push(child);
                }
            }
        }
    }
    return result;
}

std::vector<std::int64_t> NewLMTree::range_query(const float *query, const float radius, const Dataset &dataset) {
    std::vector<std::int64_t> result;
    std::queue<Node *> routes;
    routes.push(root);
    while (!routes.empty()) {
        ++node_cost;
        const auto node = routes.front();
        routes.pop();
        const auto dist_to_center = distance(query, dataset[node->center]);
        if (dist_to_center > radius + node->radius) {
            continue;
        }
        if (node->is_leaf) {
            for (auto &entry: node->items) {
                if (dist_to_center > radius + entry.first) {
                    continue;
                }
                const auto dist = distance(query, dataset[entry.second.id]);
                ++cost;
                if (dist > radius) {
                    continue;
                }
                result.emplace_back(entry.second.id);
            }
        } else {
            for (auto &child: node->items) {
                const auto dist = distance(query, dataset[child.second.child->center]);
                if (dist > radius + child.second.child->radius) {
                    continue;
                }
                routes.push(child.second.child);
            }
        }
    }
    return result;
}

void NewLMTree::insert(const std::int64_t id, Dataset &dataset, Node *node, Node *father,
                       const std::int64_t index_in_father) {
    const auto dist_to_center = distance(dataset[id], dataset[node->center]);
    node->radius = std::max(node->radius, dist_to_center);
    if (node->is_leaf) {
        node->add_data(dist_to_center, id);
    } else {
        std::int64_t closest_child_id = 0;
        float min_dist = std::numeric_limits<float>::max();
        for (std::int64_t i = 0; i < node->items.size(); ++i) {
            const auto child = node->items[i].second.child;
            if (const auto dist = distance(dataset[child->center], dataset[id]);
                dist < min_dist) {
                min_dist = dist;
                closest_child_id = i;
            }
        }
        insert(id, dataset, node->items[closest_child_id].second.child, node, closest_child_id);
    }
    if (node->is_full()) {
        auto new_node1 = new Node();
        auto new_node2 = new Node();
        if (node->is_leaf) {
            new_node1->center = node->items[0].second.id;
            float max_dist = -1;
            for (const auto &entry: node->items) {
                const auto to_split_data_id = entry.second.id;
                if (const auto dist = distance(dataset[new_node1->center], dataset[to_split_data_id]);
                    dist > max_dist) {
                    max_dist = dist;
                    new_node2->center = to_split_data_id;
                }
            }

            for (const auto &entry: node->items) {
                const auto to_split_data_id = entry.second.id;
                const auto d1 = distance(dataset[new_node1->center], dataset[to_split_data_id]);
                const auto d2 = distance(dataset[new_node2->center], dataset[to_split_data_id]);
                if (d1 == d2) {
                    if (new_node1->items.size() < new_node2->items.size()) {
                        new_node1->add_data(d1, to_split_data_id);
                    } else {
                        new_node2->add_data(d2, to_split_data_id);
                    }
                } else {
                    if (d1 < d2) {
                        new_node1->add_data(d1, to_split_data_id);
                    } else {
                        new_node2->add_data(d2, to_split_data_id);
                    }
                }
            }
            assert(!new_node1->items.empty() && !new_node2->items.empty());
            new_node1->is_leaf = new_node2->is_leaf = true;
        } else {
            new_node1->center = node->items[0].second.child->center;
            float max_dist = -1;
            for (const auto &child: node->items) {
                const auto to_split_center = child.second.child->center;
                if (const auto dist = distance(dataset[new_node1->center], dataset[to_split_center]);
                    dist > max_dist) {
                    max_dist = dist;
                    new_node2->center = to_split_center;
                }
            }

            for (auto &child: node->items) {
                auto to_split_node = child.second.child;
                const auto to_split_data_id = to_split_node->center;
                const auto d1 = distance(dataset[new_node1->center], dataset[to_split_data_id]);
                const auto d2 = distance(dataset[new_node2->center], dataset[to_split_data_id]);

                if (d1 == d2) {
                    if (new_node1->items.size() < new_node2->items.size()) {
                        new_node1->add_sub_node(d1, to_split_node);
                    } else {
                        new_node2->add_sub_node(d2, to_split_node);
                    }
                } else {
                    if (d1 < d2) {
                        new_node1->add_sub_node(d1, to_split_node);
                    } else {
                        new_node2->add_sub_node(d2, to_split_node);
                    }
                }
            }
            new_node1->is_leaf = new_node2->is_leaf = false;
        }
        delete node;
        if (father == nullptr) {
            root = new Node();
            root->is_leaf = false;
            root->center = new_node1->center;
            auto d1 = distance(dataset[root->center], dataset[new_node1->center]);
            auto d2 = distance(dataset[root->center], dataset[new_node2->center]);
            root->add_sub_node(d1, new_node1);
            root->add_sub_node(d2, new_node2);
        } else {
            auto d1 = distance(dataset[father->center], dataset[new_node1->center]);
            auto d2 = distance(dataset[father->center], dataset[new_node2->center]);
            father->set_sub_node(index_in_father, d1, new_node1);
            father->add_sub_node(d2, new_node2);
        }
    }
}
