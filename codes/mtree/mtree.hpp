#pragma once

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <map>
#include <limits>
#include "mnode.hpp"
#include "entry.hpp"

namespace MetricSpaceBenchmark::MetricIndex::MTree {

    template<typename T, int NROUTES = 4, int LEAFCAP = 50>
    class MTree {
    private:
        size_t m_count;

        MTreeNode<T, NROUTES, LEAFCAP> *m_top;

        void promote(std::vector<DBEntry<T>> &entries, RoutingObject<T> &robj1, RoutingObject<T> &robj2);

        void partition(std::vector<DBEntry<T>> &entries, RoutingObject<T> &robj1, RoutingObject<T> &robj2,
                       std::vector<DBEntry<T>> &entries1, std::vector<DBEntry<T>> &entries2);

        MTreeNode<T, NROUTES, LEAFCAP> *split(MTreeNode<T, NROUTES, LEAFCAP> *node, const Entry<T> &nobj);

        void StoreEntries(MTreeLeaf<T, NROUTES, LEAFCAP> *leaf, std::vector<DBEntry<T>> &entries);

        double calculateLowerBound(MTreeNode<T, NROUTES, LEAFCAP> *node, const T &query);

        double calculateUpperBound(MTreeNode<T, NROUTES, LEAFCAP> *node, const T &query);

        MTreeNode<T, NROUTES, LEAFCAP> *
        ChooseNode(CustomPriorityQueue<std::pair<MTreeNode<T, NROUTES, LEAFCAP> *, double>, T, NROUTES, LEAFCAP> &PR);

        void NN_Update(unsigned long k, std::vector<std::pair<long long, double>> &NN,
                       const std::pair<long long, double> &newEntry);
    public:

        MTree() : m_count(0), m_top(nullptr) {}

        void Clear();

        [[nodiscard]] size_t size() const;

        void Insert(const Entry<T> &entry);

        int DeleteEntry(const Entry<T> &entry);

        std::vector<Entry<T>> RangeQuery(T query, double radius) const;

        [[nodiscard]] size_t memory_usage() const;

        MTreeNode<T, NROUTES, LEAFCAP> *getMTop() const {
            return m_top;
        }

        void NN_Update(int k, std::vector<std::pair<long long, double>> &NN,
                       const std::pair<long long, double> &newEntry);

        std::vector<std::pair<long long, double>>
        KNN_Search(MTreeNode<T, NROUTES, LEAFCAP> *root, const T &query, int k);

    };

    template<typename T, int NROUTES, int LEAFCAP>
    void MTree<T, NROUTES, LEAFCAP>::promote(std::vector<DBEntry<T>> &entries, RoutingObject<T> &robj1,
                                             RoutingObject<T> &robj2) {

        RoutingObject<T> routes[2];

        int current = 0;
        routes[current % 2].key = entries[0].key;

        const int n_iters = 5;
        for (int i = 0; i < n_iters; i++) {
            int maxpos = -1;
            double maxd = 0;
            const int slimit = entries.size();
            for (int j = 0; j < slimit; j++) {
                double d = routes[current % 2].distance(entries[j].key);
                if (d > maxd) {
                    maxpos = j;
                    maxd = d;
                }
            }
            routes[++current % 2].key = entries[maxpos].key;
        }

        robj1.key = routes[0].key;
        robj2.key = routes[1].key;
        robj1.dis = 0;
        robj2.dis = 0;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTree<T, NROUTES, LEAFCAP>::partition(std::vector<DBEntry<T>> &entries,
                                               RoutingObject<T> &robj1, RoutingObject<T> &robj2,
                                               std::vector<DBEntry<T>> &entries1, std::vector<DBEntry<T>> &entries2) {

        double radius1 = 0;
        double radius2 = 0;
        for (int i = 0; i < (int) entries.size(); i++) {
            double d1 = robj1.distance(entries[i].key);
            double d2 = robj2.distance(entries[i].key);
            if (d1 < d2) {
                entries1.push_back({entries[i].id, entries[i].key, d1});
                if (d1 > radius1) radius1 = d1;
            } else {
                entries2.push_back({entries[i].id, entries[i].key, d2});
                if (d2 > radius2) radius2 = d2;
            }
        }

        robj1.cover_radius = radius1;
        robj2.cover_radius = radius2;
        entries.clear();
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void
    MTree<T, NROUTES, LEAFCAP>::StoreEntries(MTreeLeaf<T, NROUTES, LEAFCAP> *leaf, std::vector<DBEntry<T>> &entries) {
        while (!entries.empty()) {
            leaf->StoreEntry(entries.back());
            entries.pop_back();
        }
    }

    template<typename T, int NROUTES, int LEAFCAP>
    double MTree<T, NROUTES, LEAFCAP>::calculateLowerBound(MTreeNode<T, NROUTES, LEAFCAP> *node, const T &query) {
        if (node->isRoot()) {
            return 0;
        }
        int rdx = -1;
        MTreeNode<T, NROUTES, LEAFCAP> *parent = node->GetParentNode(rdx);
        RoutingObject<T> parentObj;
        ((MTreeInternal<T, NROUTES, LEAFCAP> *) parent)->GetRoute(rdx, parentObj);
        double distanceBetweenParentAndQuery = parentObj.distance(query);
        return std::max(distanceBetweenParentAndQuery - parentObj.cover_radius, 0.0);
    }


    template<typename T, int NROUTES, int LEAFCAP>
    double MTree<T, NROUTES, LEAFCAP>::calculateUpperBound(MTreeNode<T, NROUTES, LEAFCAP> *node, const T &query) {
        if (node->isRoot()) {
            return std::numeric_limits<double>::max();
        }

        int rdx = -1;
        MTreeNode<T, NROUTES, LEAFCAP> *parent = node->GetParentNode(rdx);
        RoutingObject<T> parentObj;
        ((MTreeInternal<T, NROUTES, LEAFCAP> *) parent)->GetRoute(rdx, parentObj);
        return parentObj.distance(query) + parentObj.cover_radius;
    }


    template<typename T, int NROUTES, int LEAFCAP>
    MTreeNode<T, NROUTES, LEAFCAP> *MTree<T, NROUTES, LEAFCAP>::ChooseNode(
            CustomPriorityQueue<std::pair<MTreeNode<T, NROUTES, LEAFCAP> *, double>, T, NROUTES, LEAFCAP> &PR) {
        auto minPair = PR.top();
        PR.pop();
        return minPair.first;
    }


    template<typename T, int NROUTES, int LEAFCAP>
    void MTree<T, NROUTES, LEAFCAP>::NN_Update(unsigned long k, std::vector<std::pair<long long, double>> &NN,
                                               const std::pair<long long, double> &newEntry) {
        auto it = std::lower_bound(NN.begin(), NN.end(), newEntry,
                                   [](const std::pair<long long, double> &a, const std::pair<long long, double> &b) {
                                       return a.second < b.second;
                                   });
        NN.insert(it, newEntry);
        if (NN.size() > k) {
            NN.pop_back();
        }
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTree<T, NROUTES, LEAFCAP>::Clear() {
        std::queue<MTreeNode<T, NROUTES, LEAFCAP> *> nodes;
        if (m_top != nullptr)
            nodes.push(m_top);

        while (!nodes.empty()) {
            MTreeNode<T, NROUTES, LEAFCAP> *current = nodes.front();
            if (typeid(*current) == typeid(MTreeInternal<T, NROUTES, LEAFCAP>)) {
                auto *internal = (MTreeInternal<T, NROUTES, LEAFCAP> *) current;
                for (int i = 0; i < NROUTES; i++) {
                    MTreeNode<T, NROUTES, LEAFCAP> *child = current->GetChildNode(i);
                    if (child) nodes.push(child);
                }
                delete internal;
            } else if (typeid(*current) == typeid(MTreeLeaf<T, NROUTES, LEAFCAP>)) {
                auto *leaf = (MTreeLeaf<T, NROUTES, LEAFCAP> *) current;
                delete leaf;
            } else {
                throw std::logic_error("no such node type");
            }
            nodes.pop();
        }
        m_count = 0;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    MTreeNode<T, NROUTES, LEAFCAP> *
    MTree<T, NROUTES, LEAFCAP>::split(MTreeNode<T, NROUTES, LEAFCAP> *node, const Entry<T> &nobj) {
        assert(typeid(*node) == typeid(MTreeLeaf<T, NROUTES, LEAFCAP>));

        auto *leaf = (MTreeLeaf<T, NROUTES, LEAFCAP> *) node;
        auto *leaf2 = new MTreeLeaf<T, NROUTES, LEAFCAP>();

        std::vector<DBEntry<T>> entries;
        leaf->GetEntries(entries);

        entries.push_back({nobj.id, nobj.key, 0});

        RoutingObject<T> robj1, robj2;
        promote(entries, robj1, robj2);

        std::vector<DBEntry<T>> entries1, entries2;
        partition(entries, robj1, robj2, entries1, entries2);
        robj1.subtree = leaf;
        robj2.subtree = leaf2;

        leaf->Clear();

        StoreEntries(leaf, entries1);
        StoreEntries(leaf2, entries2);

        MTreeInternal<T, NROUTES, LEAFCAP> *pnode;
        if (node->isRoot()) {
            auto *qnode = new MTreeInternal<T, NROUTES, LEAFCAP>();

            int rdx = qnode->StoreRoute(robj1);
            qnode->SetChildNode(leaf, rdx);

            rdx = qnode->StoreRoute(robj2);
            qnode->SetChildNode(leaf2, rdx);

            pnode = qnode;
        } else {
            int rdx;
            pnode = (MTreeInternal<T, NROUTES, LEAFCAP> *) (node->GetParentNode(rdx));
            if (pnode->isFull()) {
                auto *qnode = new MTreeInternal<T, NROUTES, LEAFCAP>();

                RoutingObject<T> pobj;
                pnode->GetRoute(rdx, pobj);

                robj1.dis = pobj.distance(robj1.key);
                int rdx1 = qnode->StoreRoute(robj1);
                qnode->SetChildNode(leaf, rdx1);

                robj2.dis = pobj.distance(robj2.key);

                int rdx2 = qnode->StoreRoute(robj2);
                qnode->SetChildNode(leaf2, rdx2);

                pnode->SetChildNode(qnode, rdx);

            } else {
                int gdx;
                auto *gnode = (MTreeInternal<T, NROUTES, LEAFCAP> *) pnode->GetParentNode(gdx);
                if (gnode != nullptr) {
                    RoutingObject<T> pobj;
                    gnode->GetRoute(gdx, pobj);
                    robj1.dis = pobj.distance(robj1.key);
                    robj2.dis = pobj.distance(robj2.key);
                }

                pnode->ConfirmRoute(robj1, rdx);
                pnode->SetChildNode(leaf, rdx);

                int rdx2 = pnode->StoreRoute(robj2);
                pnode->SetChildNode(leaf2, rdx2);
            }
        }

        return pnode;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTree<T, NROUTES, LEAFCAP>::Insert(const Entry<T> &entry) {

        MTreeNode<T, NROUTES, LEAFCAP> *node = m_top;
        if (node == nullptr) {
            auto *leaf = new MTreeLeaf<T, NROUTES, LEAFCAP>();
            DBEntry<T> dentry(entry.id, entry.key, 0);
            leaf->StoreEntry(dentry);
            m_top = leaf;
        } else {
            double d = 0;
            do {
                if (typeid(*node) == typeid(MTreeInternal<T, NROUTES, LEAFCAP>)) {
                    RoutingObject<T> robj;
                    ((MTreeInternal<T, NROUTES, LEAFCAP> *) node)->SelectRoute(entry.key, robj, true);
                    node = (MTreeNode<T, NROUTES, LEAFCAP> *) robj.subtree;
                    d = robj.key.distance(entry.key);
                } else if (typeid(*node) == typeid(MTreeLeaf<T, NROUTES, LEAFCAP>)) {
                    if (!node->isFull()) {
                        ((MTreeLeaf<T, NROUTES, LEAFCAP> *) node)->StoreEntry({entry.id, entry.key, d});
                    } else {
                        node = split(node, entry);
                        if (node->isRoot()) {
                            m_top = node;
                        }
                    }
                    node = nullptr;
                } else {
                    throw std::logic_error("no such node type");
                }
            } while (node != nullptr);
        }

        m_count += 1;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    int MTree<T, NROUTES, LEAFCAP>::DeleteEntry(const Entry<T> &entry) {
        MTreeNode<T, NROUTES, LEAFCAP> *node = m_top;

        int count = 0;
        while (node != nullptr) {
            if (typeid(*node) == typeid(MTreeInternal<T, NROUTES, LEAFCAP>)) {
                RoutingObject<T> robj;
                ((MTreeInternal<T, NROUTES, LEAFCAP> *) node)->SelectRoute(entry.key, robj, false);
                node = (MTreeNode<T, NROUTES, LEAFCAP> *) robj.subtree;
            } else if (typeid(*node) == typeid(MTreeLeaf<T, NROUTES, LEAFCAP>)) {
                auto *leaf = (MTreeLeaf<T, NROUTES, LEAFCAP> *) node;
                count = leaf->DeleteEntry(entry.key);
                node = nullptr;
            } else {
                throw std::logic_error("no such node type");
            }
        }

        m_count -= count;
        return count;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    std::vector<Entry<T>> MTree<T, NROUTES, LEAFCAP>::RangeQuery(T query, const double radius) const {
        std::vector<Entry<T>> results;
        std::queue<MTreeNode<T, NROUTES, LEAFCAP> *> nodes;

        if (m_top != nullptr)
            nodes.push(m_top);

        while (!nodes.empty()) {
            MTreeNode<T, NROUTES, LEAFCAP> *current = nodes.front();
            if (typeid(*current) == typeid(MTreeInternal<T, NROUTES, LEAFCAP>)) {
                auto *internal = (MTreeInternal<T, NROUTES, LEAFCAP> *) current;
                internal->SelectRoutes(query, radius, nodes);
            } else if (typeid(*current) == typeid(MTreeLeaf<T, NROUTES, LEAFCAP>)) {
                auto *leaf = (MTreeLeaf<T, NROUTES, LEAFCAP> *) current;
                leaf->SelectEntries(query, radius, results);
            } else {
                throw std::logic_error("no such node type");
            }
            nodes.pop();
        }
        return results;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    size_t MTree<T, NROUTES, LEAFCAP>::size() const {
        return m_count;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    size_t MTree<T, NROUTES, LEAFCAP>::memory_usage() const {
        std::queue<MTreeNode<T, NROUTES, LEAFCAP> *> nodes;
        if (m_top != nullptr)
            nodes.push(m_top);

        int n_internal = 0, n_leaf = 0, n_entry = 0;
        while (!nodes.empty()) {
            MTreeNode<T, NROUTES, LEAFCAP> *node = nodes.front();
            if (typeid(*node) == typeid(MTreeInternal<T, NROUTES, LEAFCAP>)) {
                n_internal++;
                for (int i = 0; i < NROUTES; i++) {
                    MTreeNode<T, NROUTES, LEAFCAP> *child = node->GetChildNode(i);
                    if (child) nodes.push(child);
                }
            } else if (typeid(*node) == typeid(MTreeLeaf<T, NROUTES, LEAFCAP>)) {
                n_leaf++;
                n_entry += node->size();
            }
            nodes.pop();
        }

        return (n_internal * sizeof(MTreeInternal<T, NROUTES, LEAFCAP>) +
                n_leaf * sizeof(MTreeLeaf<T, NROUTES, LEAFCAP>) +
                m_count * sizeof(DBEntry<T>) + sizeof(MTree<T, NROUTES, LEAFCAP>));
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTree<T, NROUTES, LEAFCAP>::NN_Update(int k, std::vector<std::pair<long long, double>> &NN,
                                               const std::pair<long long, double> &newEntry) {
        if (NN.size() < static_cast<size_t>(k)) {
            NN.push_back(newEntry);
            std::sort(NN.begin(), NN.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });
        } else if (newEntry.second < NN.back().second) {
            NN.back() = newEntry;
            std::sort(NN.begin(), NN.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });
        }
    }

    template<typename T, int NROUTES, int LEAFCAP>
    std::vector<std::pair<long long, double>>
    MTree<T, NROUTES, LEAFCAP>::KNN_Search(MTreeNode<T, NROUTES, LEAFCAP> *root, const T &query, int k) {
        struct QueueElement {
            MTreeNode<T, NROUTES, LEAFCAP> *node;
            double d_min;
            double d_parent_query;


            bool operator<(const QueueElement &other) const {
                return d_min > other.d_min;
            }
        };


        std::priority_queue<QueueElement> PR;

        if (root) {
            PR.push({root, 0.0, 0.0});
        }

        std::vector<std::pair<long long, double>> NN;
        for (int i = 0; i < k; ++i) {
            NN.push_back({-1, std::numeric_limits<double>::max()});
        }
        double d_k = std::numeric_limits<double>::max();

        while (!PR.empty()) {
            QueueElement elem = PR.top();
            PR.pop();

            MTreeNode<T, NROUTES, LEAFCAP> *node = elem.node;
            double d_parent_query = elem.d_parent_query;

            if (typeid(*node) == typeid(MTreeInternal<T, NROUTES, LEAFCAP>)) {
                auto *internal = static_cast<MTreeInternal<T, NROUTES, LEAFCAP> *>(node);

                for (int i = 0; i < internal->n_routeObjects; ++i) {
                    RoutingObject<T> &robj = internal->routeObjects[i];
                    if (!robj.subtree) continue;

                    if (std::abs(d_parent_query - robj.dis) <= d_k + robj.cover_radius) {

                        double d_rq = robj.distance(query);


                        if (d_rq <= d_k + robj.cover_radius) {
                            double d_min = std::max(d_rq - robj.cover_radius, 0.0);
                            double d_max = d_rq + robj.cover_radius;

                            if (d_min <= d_k) {
                                PR.push({static_cast<MTreeNode<T, NROUTES, LEAFCAP> *>(robj.subtree),
                                         d_min, d_rq});

                                if (d_max < d_k) {
                                    d_k = d_max;
                                    std::priority_queue<QueueElement> newPR;
                                    while (!PR.empty()) {
                                        if (PR.top().d_min <= d_k) {
                                            newPR.push(PR.top());
                                        }
                                        PR.pop();
                                    }
                                    PR = newPR;
                                }
                            }
                        }
                    }
                }
            } else if (typeid(*node) == typeid(MTreeLeaf<T, NROUTES, LEAFCAP>)) {
                auto *leaf = static_cast<MTreeLeaf<T, NROUTES, LEAFCAP> *>(node);

                for (auto &entry : leaf->entries) {
                    if (std::abs(d_parent_query - entry.dis) <= d_k) {
                        double d_jq = entry.distance(query);

                        if (d_jq <= d_k) {
                            NN_Update(k, NN, {entry.id, d_jq});
                            d_k = NN.back().second;

                            std::priority_queue<QueueElement> newPR;
                            while (!PR.empty()) {
                                if (PR.top().d_min <= d_k) {
                                    newPR.push(PR.top());
                                }
                                PR.pop();
                            }
                            PR = newPR;
                        }
                    }
                }
            }
        }

        return NN;
    }
}
