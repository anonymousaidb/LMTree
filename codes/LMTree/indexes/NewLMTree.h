#ifndef NEWMTREE_H
#define NEWMTREE_H
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <vector>

#include "basic.h"
#include "IndexParameters.h"

template<typename T, typename Compare>
bool is_sorted_custom(const std::vector<T> &vec, Compare comp) {
    if (vec.size() < 2) return true;

    for (size_t i = 1; i < vec.size(); ++i) {
        if (comp(vec[i], vec[i - 1])) {
            return false;
        }
    }
    return true;
}

class NewLMTree {
public:
    class Node;

    union Item {
        Node *child;
        std::int64_t id;

        explicit Item(std::int64_t _id) : id(_id) {
        }

        explicit Item(Node *_child) : child(_child) {
        }
    };

    static bool compare_node_vectors_unsorted(const std::vector<Node *> &a, const std::vector<Node *> &b);

    static bool compare_data_vectors_unsorted(const std::vector<std::int64_t> &a, const std::vector<std::int64_t> &b);

    struct Segment {
        std::int64_t pivot = -1;
        float max_dist_to_center = std::numeric_limits<float>::max();
        float min_dist_to_center = 0;
        float error_bound = 0;
        float slope = 0;
        float intercept = 0;
        std::vector<std::pair<float, Item> > items;

        Segment()
            : pivot(-1),
              max_dist_to_center(std::numeric_limits<float>::max()),
              min_dist_to_center(0),
              error_bound(0),
              slope(0),
              intercept(0),
              items() {
        }

        Segment(const Segment &) = delete;

        Segment &operator=(const Segment &) = delete;

        Segment(Segment &&) = default;

        Segment &operator=(Segment &&) = default;

        float forward(const float key) const {
            return std::round(slope * key + intercept);
        }
    };

    class Node {
    public:
        std::vector<Segment> segments;
        std::int64_t center = 0;
        float radius = 0.0f;

        std::vector<std::pair<float, Item> > items;
        float last_split_cost = -100;
        float cost = -100;
        bool is_leaf;
        short old_children_size;
        short old_seg_size;
        void update_size() {
            old_children_size = size();
        }
        void replace_child(Node *child, Node *new_child, float new_distance);

        void remove_child(Node *child);

        [[nodiscard]] std::int64_t size() const;

        [[nodiscard]] std::int64_t leaf_node_count() const;

        std::vector<Node *> leaf_children();

        std::vector<Node *> all_children() const;


        std::vector<Node *> none_pivot_children() const {
            std::vector<Node *> result;
            for (auto &seg: segments) {
                for (auto &item: seg.items) {
                    if (item.second.child->segments.size() > 1) {
                        continue;
                    }
                    result.push_back(item.second.child);
                }
            }
            return result;
        }

        void check_adding_pivot() const;

        Node() = default;

        Node(const Node &) = delete;

        Node &operator=(const Node &) = delete;

        Node(Node &&) = default;

        Node &operator=(Node &&) = default;

        void add_data(float dist_to_center, std::int64_t id);

        void add_sub_node(float dist_to_center, Node *node);

        void set_sub_node(std::int64_t index, float dist_to_center, Node *node);

        [[nodiscard]] bool is_full() const noexcept;
    };

    void try_merge(Node *node, Node *father, Dataset &dataset);
    std::size_t total_memory_usage(const Node *root) {

        std::size_t total_size = 0;
        std::queue<const Node *> q;
        q.push(root);

        while (!q.empty()) {
            const Node *node = q.front();
            q.pop();
            total_size += sizeof(Node);
            for (auto &seg:node->segments) {
                total_size += sizeof(seg);
                total_size += seg.items.size() * sizeof(seg.items.front());
            }
            if (node->is_leaf) {
                continue;
            }

            for (auto &seg:node->segments) {
                for (auto &item: seg.items) {
                    q.emplace(item.second.child);
                }
            }
        }

        return total_size;
    }
    std::size_t total_nodes_count(const Node *root) {

        std::size_t total_size = 0;
        std::queue<const Node *> q;
        q.push(root);

        while (!q.empty()) {
            const Node *node = q.front();
            q.pop();
            ++total_size;
            if (node->is_leaf) {
                continue;
            }

            for (auto &seg:node->segments) {
                for (auto &item: seg.items) {
                    q.emplace(item.second.child);
                }
            }
        }

        return total_size;
    }


    static void print_tree(const Node *node, int depth = 0);

    std::int64_t result_acc = 0;

    void merge(Node *node, Node *father, Dataset &dataset);

public:
    Node *root;

    NewLMTree() : root(new Node()) {
        root->is_leaf = true;
    }

    ~NewLMTree();

    void check_adding_pivot() const;

    static void add_pivot_model_for_leaf(Node &node, Dataset &dataset);

    static void add_pivot_model_for_inner(Node &node, Dataset &dataset);

    void add_models(Dataset &dataset);

    void merge(Dataset &dataset) {
        merge(root, nullptr, dataset);
    }

    void insert(std::int64_t id, Dataset &dataset);
    std::int64_t try_merge_times;
    void insert_with_pivot(const std::int64_t id, Dataset &dataset) {
        auto changed = insert_with_pivot(id, dataset, root, nullptr, nullptr, -1);
        if (changed && root->size() > 2 * root->old_children_size) {
            try_merge(root, nullptr, dataset);
        }
    }

    void delete_with_pivot(const std::int64_t id, Dataset &dataset) {
        delete_with_pivot(id, dataset, root, nullptr, nullptr, -1);
    }

    std::int64_t inserted_count = 0;
    bool insert_with_pivot( std::int64_t insert_id, Dataset &dataset, Node *node, Segment *father_seg,
                           Node *father_node,
                            std::int64_t index_in_father);

    void delete_with_pivot( std::int64_t delete_id, Dataset &dataset, Node *node, Segment *father_seg,
                                  Node *father_node,
                                   std::int64_t index_in_father);

    std::vector<std::int64_t> kNN_query_with_pivot(const float *query, std::int64_t k,
                                                   const Dataset &dataset) const;

    std::vector<std::int64_t> range_query_with_pivot(const float *query, float radius,
                                                     const Dataset &dataset) const;

    std::vector<std::int64_t> range_query(const float *query, float radius, const Dataset &dataset);

    void insert(std::int64_t id, Dataset &dataset, Node *node, Node *father, std::int64_t index_in_father);

    void print() const {
        print_tree(root);
    }

    std::size_t calculateNodeMemory(const Node *node) {
        return total_memory_usage(node);
    }
    std::size_t calculate_nodes_count(const Node *node) {
        return total_nodes_count(node);
    }
};


#endif
