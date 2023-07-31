#include <map>
#include <unordered_map>
#include <random>
#include <limits.h>
#include <chrono>
#include <iostream>

using namespace std;

int main() {
    map<int, int> tree_map;
    unordered_map<int, int> hash_map;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, INT_MAX);

    // Tree-map
    // 10,000 inserts into tree_map
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 10000; ++i) {
        tree_map.insert({dist(rng), dist(rng)});
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << "Run time for 10,000 insertions into tree-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // 10,000 finds into tree_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 10000; ++i) {
        tree_map.find(dist(rng));
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 10,000 finds into tree-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // 10,000 at's into tree_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 10000; ++i) {
        tree_map.count(dist(rng));
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 10,000 counts into tree-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    tree_map.clear();
    // 100,000 inserts into tree_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 100000; ++i) {
        tree_map.insert({dist(rng), dist(rng)});
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 100,000 insertions into tree-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // 100,000 finds into tree_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 100000; ++i) {
        tree_map.find(dist(rng));
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 100,000 finds into tree-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // 100,000 at's into tree_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 100000; ++i) {
        tree_map.count(dist(rng));
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 100,000 counts into tree-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // Hash-map
    // 10,000 inserts into hash_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 10000; ++i) {
        hash_map.insert({dist(rng), dist(rng)});
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 10,000 insertions into hash-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // 10,000 finds into hash_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 10000; ++i) {
        hash_map.find(dist(rng));
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 10,000 finds into hash-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // 10,000 at's into hash_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 10000; ++i) {
        hash_map.count(dist(rng));
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 10,000 counts into hash-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    hash_map.clear();
    // 100,000 inserts into hash_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 100000; ++i) {
        hash_map.insert({dist(rng), dist(rng)});
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 100,000 insertions into hash-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // 100,000 finds into hash_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 100000; ++i) {
        hash_map.find(dist(rng));
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 100,000 finds into hash-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;

    // 100,000 at's into hash_map
    start = std::chrono::steady_clock::now();
    for (size_t i{}; i < 100000; ++i) {
        hash_map.count(dist(rng));
    }
    end = std::chrono::steady_clock::now();
    cout << "Run time for 100,000 counts into hash-based map: " << std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() << " ms" << endl;  
}
