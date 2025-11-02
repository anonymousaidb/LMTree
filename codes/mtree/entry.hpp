#pragma once

#include <cstdint>
#include <algorithm>
namespace MetricSpaceBenchmark::MetricIndex::MTree {


    template<typename T>
    class Entry {
    public:
        Entry() = default;

        Entry(const long long id, const T key) : id(id), key(key) {};

        Entry(const Entry<T> &other) {
            id = other.id;
            key = other.key;
        }

        Entry<T> &operator=(const Entry<T> &other) {
            id = other.id;
            key = other.key;
            return *this;
        }

        long long id{};
        T key;
    };

    template<typename T>
    class RoutingObject {
    public:
        RoutingObject() : id(0), subtree(nullptr) {}

        RoutingObject(const long long id, const T key) : id(id), key(key), subtree(nullptr), cover_radius(0),
                                                         dis(0) {}

        RoutingObject(const RoutingObject &other) {
            id = other.id;
            key = other.key;
            subtree = other.subtree;
            cover_radius = other.cover_radius;
            dis = other.dis;
        }

        RoutingObject &operator=(const RoutingObject &other) {
            if (this != &other) {
                id = other.id;
                key = other.key;
                subtree = other.subtree;
                cover_radius = other.cover_radius;
                dis = other.dis;
            }
            return *this;
        }


        double distance(const T &other) const {
            RoutingObject<T>::n_build_ops++;
            return key.distance(other);
        }

        static unsigned long n_build_ops;
        long long id;
        T key;
        void *subtree;
        double cover_radius{};
        double dis{};
    };

    template<typename T>
    class DBEntry {
    public:
        DBEntry() = default;

        DBEntry(const long long id, const T key, const double dis) : id(id), key(key), dis(dis) {}

        DBEntry(const DBEntry &other) {
            id = other.id;
            key = other.key;
            dis = other.dis;
        }

        DBEntry &operator=(const DBEntry &other) {
            id = other.id;
            key = other.key;
            dis = other.dis;
            return *this;
        }

        double distance(const T &other) const {
            DBEntry<T>::n_query_ops++;
            return key.distance(other);
        }

        static unsigned long n_query_ops;

        long long id{};
        T key;
        double dis{};
    };

    template<typename T>
    unsigned long MTree::RoutingObject<T>::n_build_ops = 0;

    template<typename T>
    unsigned long MTree::DBEntry<T>::n_query_ops = 0;
}