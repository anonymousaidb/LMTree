#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <queue>
#include <sstream>
#include <fstream>
#include <limits>
#include <random>
#include <chrono>

using namespace std;
int dim = 2;
int objNum = 0;
int dist_comp = 0;
int knn_dist_comp = 0;
int node_access = 0;
int knn_node_access = 0;


struct KeyObject {

    static int func;
    vector<double> key;

    KeyObject() = default;
    explicit KeyObject(const vector<double>& key) : key(key) {}
    KeyObject(const KeyObject& other) = default;
    KeyObject& operator=(const KeyObject& other) = default;

    [[nodiscard]] double distance(const KeyObject& other) const {
        double dist = 0.0;
        switch (func) {
            case 2: {
                for (size_t i = 0; i < key.size(); ++i) {
                    double d = key[i] - other.key[i];
                    dist += d * d;
                }
                return std::sqrt(dist);
            }
            case 1: {
                for (size_t i = 0; i < key.size(); ++i) {
                    dist += std::abs(key[i] - other.key[i]);
                }
                return dist;
            }
            case 0: {
                for (size_t i = 0; i < key.size(); ++i) {
                    dist = std::max(dist, std::abs(key[i] - other.key[i]));
                }
                return dist;
            }
            default:
                std::cerr << "not support: " << func << std::endl;
                return -1.0;
        }
    }

    size_t size() const { return key.size(); }
    double operator[](size_t idx) const { return key[idx]; }
    double& operator[](size_t idx) { return key[idx]; }
};

int KeyObject::func = 2;


using Point = vector<double>;
using Dataset = vector<KeyObject>;
using Domain = vector<array<double,2>>;
using QuerySet = vector<Domain>;


struct Node {
    bool is_leaf;
    Domain domain;
    Dataset data;
    Node* left;
    Node* right;
    int split_dim;
    double split_value;
    int data_count;

    Node(const Domain& d, int count)
        : is_leaf(false), domain(d), left(nullptr), right(nullptr),
          split_dim(-1), split_value(0.0), data_count(count) {}
};


bool in_domain(const Point& p, const Domain& d) {
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] < d[i][0] || p[i] >= d[i][1]) return false;
    return true;
}

int count_points_in_domain(const Dataset& data, const Domain& d) {
    int cnt = 0;
    for (auto& p : data)
        if (in_domain(p.key,d)) ++cnt;
    return cnt;
}

bool intersect_query(const Domain& q, const Domain& d) {
    for (size_t i = 0; i < q.size(); ++i)
        if (q[i][1] <= d[i][0] || q[i][0] >= d[i][1]) return false;
    return true;
}

int compute_skip(const Dataset& data, const vector<pair<Domain,int>>& nodes, const QuerySet& queries) {
    int skip = 0;
    for (auto& q : queries) {
        for (auto& n : nodes) {
            if (n.second <= 0) continue;
            if (!intersect_query(q,n.first)) skip += n.second;
        }
    }
    return skip;
}

vector<pair<int,double>> generate_candidate_cut_pos(const QuerySet& queries) {
    vector<pair<int,double>> candidate_cut_pos;
    for (auto& q : queries) {
        for (size_t i = 0; i < q.size(); ++i) {
            candidate_cut_pos.push_back({(int)i,q[i][0]});
            candidate_cut_pos.push_back({(int)i,q[i][1]});
        }
    }
    return candidate_cut_pos;
}




inline double distance_pp(const Point& a, const Point& b) {
    dist_comp++;
    double val = 0.0;
    switch (KeyObject::func) {
        case 2: {
            for (size_t i = 0; i < a.size(); ++i) {
                double d = a[i] - b[i];
                val += d * d;
            }
            return std::sqrt(val);
        }
        case 1: {
            for (size_t i = 0; i < a.size(); ++i) val += std::abs(a[i] - b[i]);
            return val;
        }
        case 0: {
            for (size_t i = 0; i < a.size(); ++i) val = std::max(val, std::abs(a[i] - b[i]));
            return val;
        }
        default:
            return std::numeric_limits<double>::infinity();
    }
}

inline double min_distance_pd(const Point& a, const Domain& dom) {
    double acc = 0.0;
    double mx  = 0.0;
    dist_comp++;
    for (size_t i = 0; i < a.size(); ++i) {
        double di = 0.0;
        if (a[i] < dom[i][0]) di = dom[i][0] - a[i];
        else if (a[i] > dom[i][1]) di = a[i] - dom[i][1];
        if (KeyObject::func == 2) {
            acc += di * di;
            mx = std::max(mx, di);
        } else if (KeyObject::func == 1) {
            acc += di;
            mx = std::max(mx, di);
        } else {
            mx = std::max(mx, di);
        }
    }
    if (KeyObject::func == 2) return std::sqrt(acc);
    if (KeyObject::func == 1) return acc;
    return mx;
}


class QdTree {
public:
    QdTree(const Dataset& dataset, const QuerySet& queryset, const Domain& domains, int min_block_size)
        : data(dataset), queries(queryset), root(nullptr), min_block(min_block_size)
    {
        root = build_greedy(domains, dataset);
    }

    vector<Point> circle_range_query(const Point& center, double radius) {
        vector<Point> results;
        circle_query_qdtree(root, center, radius, results);
        return results;
    }

    vector<Point> knn_query(const Point& query_point, int k) {
        priority_queue<pair<double, Point>> max_heap;
        knn_recursive(root, query_point, k, max_heap);
        vector<Point> results;
        while (!max_heap.empty()) { results.push_back(max_heap.top().second); max_heap.pop(); }
        reverse(results.begin(), results.end());
        return results;
    }

    void insert_point(const Point& p) { insert_recursive(root, p); }
    bool  delete_point(const Point& p) { return delete_recursive(root, p); }
    bool  update_point(const Point& old_p, const Point& new_p) {
        if (delete_point(old_p)) { insert_point(new_p); return true; }
        return false;
    }

    size_t get_index_size_bytes() const {
        return compute_index_size(root);
    }



private:
    Dataset data;
    QuerySet queries;
    Node* root;
    int min_block;

    void circle_query_qdtree(Node* node, const Point& center, double radius, vector<Point>& results) {
        if (!node) return;
        node_access++;
        if (min_distance_pd(center, node->domain) > radius) return;

        if (node->is_leaf) {
            for (auto& p : node->data) {
                if (distance_pp(p.key, center) <= radius)
                    results.push_back(p.key);
            }
            return;
        }

        circle_query_qdtree(node->left, center, radius, results);
        circle_query_qdtree(node->right, center, radius, results);
    }

    void knn_recursive(Node* node, const Point& query_point, int k,
                       priority_queue<pair<double, Point>>& max_heap) {
        if (!node) return;
        knn_node_access++;
        if (node->is_leaf) {
            for (auto& p : node->data) {
                double dist = distance_pp(p.key, query_point);
                knn_dist_comp++;
                if ((int)max_heap.size() < k) {
                    max_heap.push({dist, p.key});
                } else if (dist < max_heap.top().first) {
                    max_heap.pop();
                    max_heap.push({dist, p.key});
                }
            }
            return;
        }

        double dleft  = node->left  ? min_distance_pd(query_point, node->left->domain)  : std::numeric_limits<double>::infinity();
        double dright = node->right ? min_distance_pd(query_point, node->right->domain) : std::numeric_limits<double>::infinity();

        Node* first  = dleft <= dright ? node->left  : node->right;
        Node* second = dleft <= dright ? node->right : node->left;
        double dsecond = std::min(dleft, dright) == dleft ? dright : dleft;

        knn_recursive(first, query_point, k, max_heap);

        double worst = (max_heap.size() < (size_t)k) ? std::numeric_limits<double>::infinity() : max_heap.top().first;
        if (second && dsecond < worst) {
            knn_recursive(second, query_point, k, max_heap);
        }
    }

    void insert_recursive(Node* node, const Point& p) {
        if (node->is_leaf) {
            node->data.push_back(KeyObject(p));
            node->data_count++;
            return;
        }
        int sd = node->split_dim;
        if (p[sd] < node->split_value) insert_recursive(node->left, p);
        else insert_recursive(node->right, p);
        node->data_count++;
    }

    bool delete_recursive(Node* node, const Point& p) {
        if (node->is_leaf) {
            auto it = find_if(node->data.begin(), node->data.end(),
                              [&](const KeyObject& k){ return k.key == p; });
            if (it != node->data.end()) {
                node->data.erase(it);
                node->data_count--;
                return true;
            }
            return false;
        }
        int sd = node->split_dim;
        bool ok = (p[sd] < node->split_value)
                  ? delete_recursive(node->left, p)
                  : delete_recursive(node->right, p);
        if (ok) node->data_count--;
        return ok;
    }

    Node* build_greedy(const Domain& domain, const Dataset& subset) {
        Node* node = new Node(domain, (int)subset.size());
        if ((int)subset.size() <= 2 * min_block) {
            node->is_leaf = true;
            node->data = subset;
            return node;
        }

        auto candidate_cut_pos = generate_candidate_cut_pos(queries);
        int max_skip = -1; int best_dim = -1; double best_split = 0.0;
        for (auto& cand : candidate_cut_pos) {
            int d = cand.first; double val = cand.second;
            Domain left = domain, right = domain;
            left[d][1] = val; right[d][0] = val;
            int left_count  = count_points_in_domain(subset, left);
            int right_count = count_points_in_domain(subset, right);
            if (left_count < min_block || right_count < min_block) continue;
            vector<pair<Domain,int>> temp_nodes = { {left,left_count}, {right,right_count} };
            int skip = compute_skip(data, temp_nodes, queries);
            if (skip > max_skip) { max_skip = skip; best_dim = d; best_split = val; }
        }
        if (max_skip <= 0) {
            node->is_leaf = true;
            node->data = subset;
            return node;
        }

        Domain left_domain = domain, right_domain = domain;
        left_domain[best_dim][1]  = best_split;
        right_domain[best_dim][0] = best_split;

        Dataset left_data, right_data;
        left_data.reserve(subset.size());
        right_data.reserve(subset.size());
        for (auto& p : subset) {
            (p[best_dim] < best_split ? left_data : right_data).push_back(p);
        }

        node->split_dim = best_dim;
        node->split_value = best_split;
        node->left  = build_greedy(left_domain, left_data);
        node->right = build_greedy(right_domain, right_data);
        return node;
    }


    size_t compute_index_size(Node* node) const{
        if (!node) return 0;

        size_t size = sizeof(Node);

        size += node->domain.size() * 2 * sizeof(double);

        if (node->is_leaf) {
            size += node->data.size() * sizeof(KeyObject);
            for (auto& p : node->data) {
                size += p.key.size() * sizeof(double);
            }
        }

        size += compute_index_size(node->left);
        size += compute_index_size(node->right);

        return size;
    }

};

void read_data_from_file(int type, const string &filename, vector<KeyObject> &entries) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    if (type == 0) {
        file >> dim >> objNum >> KeyObject::func;
        string line; getline(file, line);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> data;
        double value;
        while (iss >> value) data.push_back(value);
        if (!data.empty()) entries.emplace_back(KeyObject(data));
    }
    file.close();
}


void read_data_from_file_1(int type, const string &filename, vector<KeyObject> &entries) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    if (type == 0) {
        file >> dim >> objNum >> KeyObject::func;
        string line;
        getline(file, line);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::vector<double> data;
        double value;
        int cnt = 0;

        while (cnt < dim && (iss >> value)) {
            data.push_back(value);
            cnt++;
        }

        if (!data.empty()) entries.emplace_back(KeyObject(data));
    }

    file.close();
}






Domain compute_domains(const vector<KeyObject>& data, int dim) {
    Domain domains(dim, {numeric_limits<double>::max(), numeric_limits<double>::lowest()});
    for (auto& obj : data) {
        for (int i = 0; i < dim; ++i) {
            if (obj.key[i] < domains[i][0]) domains[i][0] = obj.key[i];
            if (obj.key[i] > domains[i][1]) domains[i][1] = obj.key[i];
        }
    }
    return domains;
}

QuerySet generate_random_queries(const Dataset& entries, int num_queries, double range_ratio = 0.2) {
    QuerySet queries;
    if (entries.empty()) return queries;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, (int)entries.size()-1);

    for (int q = 0; q < num_queries; ++q) {
        const auto& sample = entries[dis(gen)];
        Domain dom;
        for (int d = 0; d < dim; ++d) {
            double val = sample.key[d];
            double offset = range_ratio * val;
            dom.push_back({val - offset, val + offset});
        }
        queries.push_back(dom);
    }
    return queries;
}

vector<Domain> generate_queries_from_points(const Dataset& data, double radius) {
    vector<Domain> queries;
    for (auto& obj : data) {
        Domain q;
        for (size_t i = 0; i < obj.key.size(); ++i) {
            q.push_back({obj.key[i]-radius, obj.key[i]+radius});
        }
        queries.push_back(q);
    }
    return queries;
}

int main() {
    KeyObject::func = 2;
    string dataFileName = "../../dataset/LA_example.txt";
    string queryFileName = "../../dataset/LA_query.txt";
    std::cout<<"dataFileName" << dataFileName << std::endl;
    vector<KeyObject> entries;
    read_data_from_file(0, dataFileName, entries);
    Domain domain = compute_domains(entries, dim);

    vector<KeyObject> query_points;
    read_data_from_file(1, queryFileName, query_points);
    double radius[5] = {989, 1409, 1875, 2314, 3090};
    QuerySet queries = generate_queries_from_points(query_points, radius[2]);


    QdTree tree(entries, queries, domain, 256);

    for(auto& rq:radius) {
        for (auto& q : query_points) {
            auto res = tree.circle_range_query(q.key, rq);
        }
    }

    int k_array[5] = {20, 50, 100, 150, 200};
    for (auto&k:k_array) {
        for (auto& q : query_points) {
            auto knn_res = tree.knn_query(q.key, k);
        }
    }

    return 0;
}
