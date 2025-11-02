#pragma once

#include <cstdlib>
#include <fstream>
#include <ctime>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <queue>
#include <cfloat>
#include <algorithm>

extern int dim;
extern int func;
extern int objNum;
extern double compdists;
using namespace std;

struct DataObject {
    vector<float> data;

    explicit DataObject(vector<float> d) : data(move(d)) {}
};

class Node {
public:
    bool isLeaf;

    virtual ~Node() = default;

    virtual size_t memoryUsage() const = 0;
};

class LeafNode : public Node {
public:
    vector<DataObject> objects;
    vector<double> dist_to_parent;

    LeafNode() { isLeaf = true; }

    size_t memoryUsage() const override {
        size_t total = sizeof(*this);
        total += objects.size() * sizeof(DataObject);
        for (const auto &obj: objects) {
            total += obj.data.size() * sizeof(float);
        }
        total += dist_to_parent.size() * sizeof(double);
        return total;
    }
};

class InternalNode : public Node {
public:
    vector<DataObject> pivots;
    vector<shared_ptr<Node>> children;
    vector<vector<double>> max_dist;
    vector<vector<double>> min_dist;

    InternalNode() { isLeaf = false; }

    size_t memoryUsage() const override {
        size_t total = sizeof(*this);
        total += pivots.size() * sizeof(DataObject);
        for (const auto &pivot: pivots) {
            total += pivot.data.size() * sizeof(float);
        }

        for (const auto &vec: max_dist) {
            total += vec.size() * sizeof(double);
        }
        for (const auto &vec: min_dist) {
            total += vec.size() * sizeof(double);
        }

        for (const auto &child: children) {
            if (child) total += child->memoryUsage();
        }
        return total;
    }
};

class EGNAT {
private:
    shared_ptr<Node> root;
    int leafObjCnt;
    int internalObjCnt;

public:
    EGNAT() : root(nullptr) {}

    size_t getMemoryUsage() const {
        return root ? root->memoryUsage() : 0;
    }

    void build(char *dataFileName);

    void insert(shared_ptr<Node> &node, const DataObject &obj, double dist);

    void insert(const DataObject &obj) {
        if (!root) {

            auto leaf = make_shared<LeafNode>();
            leaf->objects.push_back(obj);
            leaf->dist_to_parent.push_back(0.0);
            root = leaf;
        } else {

            insert(root, obj, 0.0);
        }
    }

    void bulkLoad(char *dataFileName);

    shared_ptr<Node> _bulkLoad(vector<DataObject> &objs);

    double distance(const DataObject &data1, const DataObject &data2);

    int rangeSearch(float *query, double radius);

    void _rangeSearch(shared_ptr<Node> node, vector<float> &query, double radius, double p_q_dist, int &count);

    double knnSearch(float *query, int k);

    void _knnSearch(shared_ptr<Node> node, vector<float> &query, int k, vector<pair<double, DataObject>> &resultSet,
                    double p_q_dist);
};


void EGNAT::build(char *dataFileName) {
    ifstream dataFile(dataFileName, ios::in);
    dataFile >> dim >> objNum >> func;

    leafObjCnt = 1024;
    internalObjCnt = 218;

    vector<DataObject> allObjects;
    allObjects.reserve(objNum);

    for (int i = 0; i < objNum; i++) {
        vector<float> objData(dim);
        for (int j = 0; j < dim; j++) dataFile >> objData[j];
        allObjects.emplace_back(objData);
    }
    dataFile.close();

    root = _bulkLoad(allObjects);
}

shared_ptr<Node> EGNAT::_bulkLoad(vector<DataObject> &objs) {
    if (objs.size() <= leafObjCnt) {
        auto leaf = make_shared<LeafNode>();
        leaf->objects = move(objs);
        leaf->dist_to_parent.resize(leaf->objects.size(), 0.0);
        return leaf;
    }

    auto internal = make_shared<InternalNode>();

    srand((unsigned) time(NULL));
    vector<int> pivotIndices;
    while (pivotIndices.size() < internalObjCnt) {
        int index = rand() % objs.size();
        if (find(pivotIndices.begin(), pivotIndices.end(), index) == pivotIndices.end()) {
            pivotIndices.push_back(index);
            internal->pivots.push_back(objs[index]);
        }
    }

    internal->max_dist.resize(internalObjCnt, vector<double>(internalObjCnt, 0.0));
    internal->min_dist.resize(internalObjCnt, vector<double>(internalObjCnt, DBL_MAX));

    for (int i = 0; i < internalObjCnt; i++) {
        for (int j = i + 1; j < internalObjCnt; j++) {
            double dist = distance(internal->pivots[i], internal->pivots[j]);
            internal->max_dist[i][j] = internal->max_dist[j][i] = dist;
            internal->min_dist[i][j] = internal->min_dist[j][i] = dist;
        }
    }

    vector<vector<DataObject>> groups(internalObjCnt);
    for (size_t i = 0; i < objs.size(); i++) {
        if (find(pivotIndices.begin(), pivotIndices.end(), i) != pivotIndices.end())
            continue;

        double minDist = DBL_MAX;
        int minId = -1;
        for (int j = 0; j < internalObjCnt; j++) {
            double dist = distance(objs[i], internal->pivots[j]);
            if (dist < minDist) {
                minDist = dist;
                minId = j;
            }
        }

        groups[minId].push_back(objs[i]);

        for (int j = 0; j < internalObjCnt; j++) {
            double dist = distance(objs[i], internal->pivots[j]);
            internal->max_dist[j][minId] = max(internal->max_dist[j][minId], dist);
            internal->min_dist[j][minId] = min(internal->min_dist[j][minId], dist);
        }
    }

    for (int i = 0; i < internalObjCnt; i++) {
        if (!groups[i].empty()) {
            internal->children.push_back(_bulkLoad(groups[i]));
        } else {
            internal->children.push_back(nullptr);
        }
    }

    return internal;
}

void EGNAT::insert(shared_ptr<Node> &node, const DataObject &obj, double dist) {
    if (!node) {
        auto leaf = make_shared<LeafNode>();
        leaf->objects.push_back(obj);
        leaf->dist_to_parent.push_back(dist);
        node = leaf;
        return;
    }

    if (node->isLeaf) {
        auto leaf = dynamic_pointer_cast<LeafNode>(node);
        if (leaf->objects.size() < leafObjCnt) {
            leaf->objects.push_back(obj);
            leaf->dist_to_parent.push_back(dist);
        } else {
            auto internal = make_shared<InternalNode>();

            vector<DataObject> allObjs = leaf->objects;
            allObjs.push_back(obj);
            vector<double> allDists = leaf->dist_to_parent;
            allDists.push_back(dist);

            srand((unsigned) time(NULL));
            vector<int> pivotIndices;
            while (pivotIndices.size() < internalObjCnt) {
                int index = rand() % allObjs.size();
                if (find(pivotIndices.begin(), pivotIndices.end(), index) == pivotIndices.end()) {
                    pivotIndices.push_back(index);
                    internal->pivots.push_back(allObjs[index]);
                }
            }

            internal->max_dist.resize(internalObjCnt, vector<double>(internalObjCnt, 0.0));
            internal->min_dist.resize(internalObjCnt, vector<double>(internalObjCnt, DBL_MAX));

            for (int i = 0; i < internalObjCnt; i++) {
                for (int j = i + 1; j < internalObjCnt; j++) {
                    double dist = distance(internal->pivots[i], internal->pivots[j]);
                    internal->max_dist[i][j] = internal->max_dist[j][i] = dist;
                    internal->min_dist[i][j] = internal->min_dist[j][i] = dist;
                }
            }

            vector<vector<DataObject>> groups(internalObjCnt);
            vector<vector<double>> groupDists(internalObjCnt);

            for (size_t i = 0; i < allObjs.size(); i++) {
                double minDist = DBL_MAX;
                int minId = -1;
                for (int j = 0; j < internalObjCnt; j++) {
                    double dist = distance(allObjs[i], internal->pivots[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        minId = j;
                    }
                }

                groups[minId].push_back(allObjs[i]);
                groupDists[minId].push_back(minDist);

                for (int j = 0; j < internalObjCnt; j++) {
                    double dist = distance(allObjs[i], internal->pivots[j]);
                    internal->max_dist[j][minId] = max(internal->max_dist[j][minId], dist);
                    internal->min_dist[j][minId] = min(internal->min_dist[j][minId], dist);
                }
            }

            for (int i = 0; i < internalObjCnt; i++) {
                if (!groups[i].empty()) {
                    auto childLeaf = make_shared<LeafNode>();
                    childLeaf->objects = move(groups[i]);
                    childLeaf->dist_to_parent = move(groupDists[i]);
                    internal->children.push_back(childLeaf);
                } else {
                    internal->children.push_back(nullptr);
                }
            }

            node = internal;
        }
    } else {
        auto internal = dynamic_pointer_cast<InternalNode>(node);

        double minDist = DBL_MAX;
        int minId = -1;
        for (int i = 0; i < internal->pivots.size(); i++) {
            double dist = distance(obj, internal->pivots[i]);
            if (dist < minDist) {
                minDist = dist;
                minId = i;
            }
        }

        for (int i = 0; i < internal->pivots.size(); i++) {
            double dist = distance(obj, internal->pivots[i]);
            internal->max_dist[i][minId] = max(internal->max_dist[i][minId], dist);
            internal->min_dist[i][minId] = min(internal->min_dist[i][minId], dist);
        }

        insert(internal->children[minId], obj, minDist);
    }
}

void EGNAT::bulkLoad(char *dataFileName) {
    build(dataFileName);
}

double EGNAT::distance(const DataObject &data1, const DataObject &data2) {
    compdists++;
    double tot = 0, dif;
    if (func == 1) {
        for (int i = 0; i < dim; i++) {
            dif = (data1.data[i] - data2.data[i]);
            if (dif < 0) dif = -dif;
            tot += dif;
        }
    } else if (func == 2) {
        for (int i = 0; i < dim; i++) {
            tot += pow(data1.data[i] - data2.data[i], 2);
        }
        tot = sqrt(tot);
    } else {
        double max = 0;
        for (int i = 0; i < dim; i++) {
            dif = (data1.data[i] - data2.data[i]);
            if (dif < 0) dif = -dif;
            if (dif > max) max = dif;
        }
        tot = max;
    }
    return tot;
}

int EGNAT::rangeSearch(float *query, double radius) {
    vector<float> queryVec(query, query + dim);
    int count = 0;
    _rangeSearch(root, queryVec, radius, 0.0, count);
    return count;
}

void EGNAT::_rangeSearch(shared_ptr<Node> node, vector<float> &query, double radius, double p_q_dist, int &count) {
    if (!node) return;

    if (node->isLeaf) {
        auto leaf = dynamic_pointer_cast<LeafNode>(node);
        for (size_t i = 0; i < leaf->objects.size(); i++) {
            if (fabs(leaf->dist_to_parent[i] - p_q_dist) <= radius) {
                DataObject obj(query);
                double dist = distance(leaf->objects[i], obj);
                if (dist <= radius) count++;
            }
        }
    } else {
        auto internal = dynamic_pointer_cast<InternalNode>(node);

        vector<double> dists(internal->pivots.size());
        double minDist = DBL_MAX;
        int minId = -1;

        DataObject queryObj(query);
        for (size_t i = 0; i < internal->pivots.size(); i++) {
            dists[i] = distance(internal->pivots[i], queryObj);
            if (dists[i] < minDist) {
                minDist = dists[i];
                minId = i;
            }
        }

        for (size_t i = 0; i < internal->pivots.size(); i++) {
            int min = dists[minId] - radius;
            int max = dists[minId] + radius;
            int sum = 2 * radius;
            int sum2 = internal->max_dist[minId][i] - internal->min_dist[minId][i];
            int upper = max > internal->max_dist[minId][i] ? max : internal->max_dist[minId][i];
            int lower = min < internal->min_dist[minId][i] ? min : internal->min_dist[minId][i];

            if ((upper - lower) <= (sum2 + sum)) {
                if (dists[i] <= radius) count++;
                if (internal->children[i]) {
                    _rangeSearch(internal->children[i], query, radius, dists[i], count);
                }
            }
        }
    }
}

double EGNAT::knnSearch(float *query, int k) {
    vector<float> queryVec(query, query + dim);
    vector<pair<double, DataObject>> resultSet;

    auto cmp = [](const pair<double, DataObject> &a, const pair<double, DataObject> &b) {
        return a.first < b.first;
    };

    _knnSearch(root, queryVec, k, resultSet, 0.0);

    sort(resultSet.begin(), resultSet.end(), cmp);

    if (resultSet.size() >= k) {
        return resultSet[k - 1].first;
    } else {
        return -1.0;
    }
}

void
EGNAT::_knnSearch(shared_ptr<Node> node, vector<float> &query, int k, vector<pair<double, DataObject>> &resultSet,
                  double p_q_dist) {
    if (!node) return;

    DataObject queryObj(query);

    if (node->isLeaf) {
        auto leaf = dynamic_pointer_cast<LeafNode>(node);
        for (size_t i = 0; i < leaf->objects.size(); i++) {
            if (resultSet.size() < k || fabs(leaf->dist_to_parent[i] - p_q_dist) <= resultSet.back().first) {
                double dist = distance(leaf->objects[i], queryObj);
                if (resultSet.size() < k || dist < resultSet.back().first) {
                    resultSet.emplace_back(dist, leaf->objects[i]);
                    sort(resultSet.begin(), resultSet.end(),
                         [](const pair<double, DataObject> &a, const pair<double, DataObject> &b) {
                             return a.first > b.first;
                         });
                    if (resultSet.size() > k) {
                        resultSet.pop_back();
                    }
                }
            }
        }
    } else {
        auto internal = dynamic_pointer_cast<InternalNode>(node);

        vector<double> dists(internal->pivots.size());
        double minDist = DBL_MAX;
        int minId = -1;

        for (size_t i = 0; i < internal->pivots.size(); i++) {
            dists[i] = distance(internal->pivots[i], queryObj);
            if (dists[i] < minDist) {
                minDist = dists[i];
                minId = i;
            }
        }

        for (size_t i = 0; i < internal->pivots.size(); i++) {
            int min = dists[minId] - (resultSet.size() < k ? DBL_MAX : resultSet.back().first);
            int max = dists[minId] + (resultSet.size() < k ? DBL_MAX : resultSet.back().first);
            int sum = 2 * (resultSet.size() < k ? DBL_MAX : resultSet.back().first);
            int sum2 = internal->max_dist[minId][i] - internal->min_dist[minId][i];
            int upper = max > internal->max_dist[minId][i] ? max : internal->max_dist[minId][i];
            int lower = min < internal->min_dist[minId][i] ? min : internal->min_dist[minId][i];

            if (resultSet.size() < k || (upper - lower) <= (sum2 + sum)) {
                if (resultSet.size() < k || dists[i] < resultSet.back().first) {
                    resultSet.emplace_back(dists[i], internal->pivots[i]);
                    sort(resultSet.begin(), resultSet.end(),
                         [](const pair<double, DataObject> &a, const pair<double, DataObject> &b) {
                             return a.first > b.first;
                         });
                    if (resultSet.size() > k) {
                        resultSet.pop_back();
                    }
                }
                if (internal->children[i]) {
                    _knnSearch(internal->children[i], query, k, resultSet, dists[i]);
                }
            }
        }
    }
}
