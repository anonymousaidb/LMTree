#pragma once

#include <vector>
#include <array>
#include <queue>
#include <cfloat>
#include <cassert>
#include "entry.hpp"
#include <stdexcept>

namespace MetricSpaceBenchmark::MetricIndex::MTree {

    template<typename T, int NROUTES = 4, int LEAFCAP = 50>
    class MTreeNode {
    protected:
        MTreeNode<T, NROUTES, LEAFCAP> *parent;
        int rindex{};

    public:
        MTreeNode() : parent(nullptr) {}

        virtual ~MTreeNode() = default;

        [[nodiscard]] virtual int size() const = 0;

        [[nodiscard]] virtual bool isFull() const = 0;

        [[nodiscard]] bool isRoot() const;

        MTreeNode<T, NROUTES, LEAFCAP> *GetParentNode(int &rdx) const;

        void SetParentNode(MTreeNode<T, NROUTES, LEAFCAP> *parentNode, int rdx);

        virtual void SetChildNode(MTreeNode<T, NROUTES, LEAFCAP> *child, int rdx) = 0;

        virtual MTreeNode<T, NROUTES, LEAFCAP> *GetChildNode(int rdx) const = 0;

        virtual void Clear() = 0;
    };

    template<typename T, int NROUTES, int LEAFCAP>
    class MTreeInternal : public MTreeNode<T, NROUTES, LEAFCAP> {
    public:
        int n_routeObjects;
        RoutingObject<T> routeObjects[NROUTES];

    public:
        MTreeInternal();

        ~MTreeInternal() = default;

        [[nodiscard]] int size() const;

        [[nodiscard]] bool isFull() const;


        void GetRoutes(std::vector<RoutingObject<T>> &Routes) const;


        int SelectRoute(T nobj, RoutingObject<T> &robj, bool insert);


        void SelectRoutes(T query, double radius, std::queue<MTreeNode<T, NROUTES, LEAFCAP> *> &nodes) const;

        int StoreRoute(const RoutingObject<T> &robj);

        void ConfirmRoute(const RoutingObject<T> &robj, int rdx);

        void GetRoute(int rdx, RoutingObject<T> &route);

        void SetChildNode(MTreeNode<T, NROUTES, LEAFCAP> *child, int rdx);

        MTreeNode<T, NROUTES, LEAFCAP> *GetChildNode(int rdx) const;

        void Clear();

    };

    template<typename T, int NROUTES, int LEAFCAP>
    class MTreeLeaf : public MTreeNode<T, NROUTES, LEAFCAP> {
    public:
        std::vector<DBEntry<T>> entries;

    public:
        MTreeLeaf() = default;

        ~MTreeLeaf();

        [[nodiscard]] int size() const;

        [[nodiscard]] bool isFull() const;

        int StoreEntry(const DBEntry<T> &nobj);

        void GetEntries(std::vector<DBEntry<T>> &dbentries) const;

        void SelectEntries(T query, double radius, std::vector<Entry<T>> &results) const;

        int DeleteEntry(const T &entry);

        void SetChildNode(MTreeNode<T, NROUTES, LEAFCAP> *child, int rdx);

        MTreeNode<T, NROUTES, LEAFCAP> *GetChildNode(int rdx) const;

        void Clear();
    };

    template<typename Element, typename T, int NROUTES, int LEAFCAP>
    class CustomPriorityQueue {
    private:
        std::vector<Element> heap;
    public:
        void push(const Element &e) {
            heap.push_back(e);
            std::push_heap(heap.begin(), heap.end(), std::greater<Element>());
        }

        typename std::vector<Element>::iterator find(const Element &target) {
            return std::find(heap.begin(), heap.end(), target);
        }

        const Element &top() const {
            if (heap.empty()) {
                throw std::out_of_range("Heap is empty");
            }
            return heap.front();
        }

        [[nodiscard]] bool empty() const {
            return heap.empty();
        }

        void erase(const Element &target) {
            auto it = find(target);
            if (it != heap.end()) {
                *it = heap.back();
                heap.pop_back();
                std::make_heap(heap.begin(), heap.end(), std::greater<Element>());
            }
        }

        void pop() {
            if (heap.empty()) {
                throw std::out_of_range("Heap is empty");
            }
            std::pop_heap(heap.begin(), heap.end(), std::greater<Element>());
            heap.pop_back();
        }

        typename std::vector<Element>::iterator begin() {
            return heap.begin();
        }

        typename std::vector<Element>::iterator end() {
            return heap.end();
        }
    };

    template<typename T, int NROUTES, int LEAFCAP>
    bool MTreeNode<T, NROUTES, LEAFCAP>::isRoot() const {
        return (parent == nullptr);
    }

    template<typename T, int NROUTES, int LEAFCAP>
    MTreeNode<T, NROUTES, LEAFCAP> *MTreeNode<T, NROUTES, LEAFCAP>::GetParentNode(int &rdx) const {
        rdx = rindex;
        return parent;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeNode<T, NROUTES, LEAFCAP>::SetParentNode(MTreeNode<T, NROUTES, LEAFCAP> *parentNode, const int rdx) {
        assert(rdx >= 0 && rdx < NROUTES);
        parent = parentNode;
        rindex = rdx;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    MTreeInternal<T, NROUTES, LEAFCAP>::MTreeInternal() {
        n_routeObjects = 0;
        for (int i = 0; i < NROUTES; i++) {
            routeObjects[i].subtree = nullptr;
        }
    }

    template<typename T, int NROUTES, int LEAFCAP>
    int MTreeInternal<T, NROUTES, LEAFCAP>::size() const {
        return n_routeObjects;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    bool MTreeInternal<T, NROUTES, LEAFCAP>::isFull() const {
        return (n_routeObjects >= NROUTES);
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeInternal<T, NROUTES, LEAFCAP>::GetRoutes(std::vector<RoutingObject<T>> &Routes) const {
        for (int i = 0; i < NROUTES; i++) {
            if (routeObjects[i].subtree != nullptr)
                Routes.push_back(routeObjects[i]);
        }
    }


    template<typename T, int NROUTES, int LEAFCAP>
    int MTreeInternal<T, NROUTES, LEAFCAP>::SelectRoute(const T nobj, RoutingObject<T> &robj, bool insert) {
        int min_pos = -1;
        auto min_dist = DBL_MAX;
        for (int i = 0; i < NROUTES; i++) {
            if (routeObjects[i].subtree != nullptr) {
                const double d = routeObjects[i].distance(nobj);
                if (d < min_dist) {
                    min_pos = i;
                    min_dist = d;
                }
            }
        }

        if (min_pos < 0)
            throw std::logic_error("unable to find route entry");


        if (insert && min_dist > routeObjects[min_pos].cover_radius)
            routeObjects[min_pos].cover_radius = min_dist;

        robj = routeObjects[min_pos];

        return min_pos;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeInternal<T, NROUTES, LEAFCAP>::SelectRoutes(const T query, const double radius,
                                                          std::queue<MTreeNode<T, NROUTES, LEAFCAP> *> &nodes) const {

        double d = 0;
        if (this->parent != nullptr) {
            RoutingObject<T> pobj;
            ((MTreeInternal<T, NROUTES, LEAFCAP> *) this->parent)->GetRoute(this->rindex, pobj);
            d = pobj.distance(query);
        }

        for (int i = 0; i < NROUTES; i++) {
            if (routeObjects[i].subtree != nullptr) {
                if (abs(d - routeObjects[i].dis) <= radius + routeObjects[i].cover_radius) {
                    if (routeObjects[i].distance(query) <=
                        radius + routeObjects[i].cover_radius) {
                        nodes.push((MTreeNode<T, NROUTES, LEAFCAP> *) routeObjects[i].subtree);
                    }
                }
            }
        }
    }

    template<typename T, int NROUTES, int LEAFCAP>
    int MTreeInternal<T, NROUTES, LEAFCAP>::StoreRoute(const RoutingObject<T> &robj) {
        assert(n_routeObjects < NROUTES);

        int index = -1;
        for (int i = 0; i < NROUTES; i++) {
            if (routeObjects[i].subtree == nullptr) {
                routeObjects[i] = robj;
                index = i;
                n_routeObjects++;
                break;
            }
        }
        return index;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeInternal<T, NROUTES, LEAFCAP>::ConfirmRoute(const RoutingObject<T> &robj, const int rdx) {
        assert(rdx >= 0 && rdx < NROUTES && robj.subtree != nullptr);
        routeObjects[rdx] = robj;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeInternal<T, NROUTES, LEAFCAP>::GetRoute(const int rdx, RoutingObject<T> &route) {
        assert(rdx >= 0 && rdx < NROUTES);
        route = routeObjects[rdx];
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeInternal<T, NROUTES, LEAFCAP>::SetChildNode(MTreeNode<T, NROUTES, LEAFCAP> *child, int rdx) {
        assert(rdx >= 0 && rdx < NROUTES);
        routeObjects[rdx].subtree = child;
        child->SetParentNode(this, rdx);
    }

    template<typename T, int NROUTES, int LEAFCAP>
    MTreeNode<T, NROUTES, LEAFCAP> *
    MTreeInternal<T, NROUTES, LEAFCAP>::GetChildNode(int rdx) const {
        assert(rdx >= 0 && rdx < NROUTES);
        return (MTreeNode<T, NROUTES, LEAFCAP> *) routeObjects[rdx].subtree;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeInternal<T, NROUTES, LEAFCAP>::Clear() {
        n_routeObjects = 0;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    MTreeLeaf<T, NROUTES, LEAFCAP>::MTreeLeaf::~MTreeLeaf() {
        entries.clear();
    }

    template<typename T, int NROUTES, int LEAFCAP>
    int MTreeLeaf<T, NROUTES, LEAFCAP>::size() const {
        return entries.size();
    }

    template<typename T, int NROUTES, int LEAFCAP>
    bool MTreeLeaf<T, NROUTES, LEAFCAP>::isFull() const {
        return (entries.size() >= LEAFCAP);
    }

    template<typename T, int NROUTES, int LEAFCAP>
    int MTreeLeaf<T, NROUTES, LEAFCAP>::StoreEntry(const DBEntry<T> &nobj) {
        if (entries.size() >= LEAFCAP)
            throw std::out_of_range("full leaf node");

        int index = entries.size();
        entries.push_back(nobj);
        return index;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeLeaf<T, NROUTES, LEAFCAP>::GetEntries(std::vector<DBEntry<T>> &dbentries) const {
        for (auto &e: entries) {
            dbentries.push_back(e);
        }
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeLeaf<T, NROUTES, LEAFCAP>::SelectEntries(const T query, const double radius,
                                                       std::vector<Entry<T>> &results) const {
        double d = 0;
        if (this->parent != nullptr) {
            RoutingObject<T> pobj;
            ((MTreeInternal<T, NROUTES, LEAFCAP> *) this->parent)->GetRoute(this->rindex, pobj);
            d = pobj.distance(query);
        }

        for (int j = 0; j < (int) entries.size(); j++) {
            if (abs(d - entries[j].dis) <= radius) {
                if (entries[j].distance(query) <= radius) {
                    results.push_back({entries[j].id, entries[j].key});
                }
            }
        }
    }

    template<typename T, int NROUTES, int LEAFCAP>
    int MTreeLeaf<T, NROUTES, LEAFCAP>::DeleteEntry(const T &entry) {
        int count = 0;

        double d = 0;
        if (this->parent != nullptr) {
            RoutingObject<T> pobj;
            ((MTreeInternal<T, NROUTES, LEAFCAP> *) this->parent)->GetRoute(this->rindex, pobj);
            d = pobj.key.distance(entry.key);
        }

        for (int j = 0; j < (int) entries.size(); j++) {
            if (d == entries[j].dis) {
                if (entry.distance(entries[j].key) == 0) {
                    entries[j] = entries.back();
                    entries.pop_back();
                    count++;
                }
            }
        }

        return count;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeLeaf<T, NROUTES, LEAFCAP>::SetChildNode(MTreeNode<T, NROUTES, LEAFCAP> *child, const int rdx) {
    }

    template<typename T, int NROUTES, int LEAFCAP>
    MTreeNode<T, NROUTES, LEAFCAP> *MTreeLeaf<T, NROUTES, LEAFCAP>::GetChildNode(const int rdx) const {
        return nullptr;
    }

    template<typename T, int NROUTES, int LEAFCAP>
    void MTreeLeaf<T, NROUTES, LEAFCAP>::Clear() {
        entries.clear();
    }
}


