#include <iostream>
#include <random>
#include <thread>
#include <vector>

// g++ ex5.cpp -o ex5.out -pthread -std=c++11
// nice -n 19 ./ex5.out 1414

void random_func(int id, int arg) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(0, 9999);
    int rand_num;
    do {
        rand_num = distr(generator);
    } while (rand_num != arg);
    std::cout << "Thread " << id << " completed." << std::endl;
}

int main(int argc, char* argv[]) {
    // Validate args
    if (argc != 2) {
        std::cout << "Error: Usage, invalid number of command line arguments" << std::endl;
        return -1;
    }

    int cmd_arg = std::stoi(argv[1]);
    std::vector<std::thread> threads;
    
    // Start threads
    for (int i=0; i<10; ++i) {
        threads.push_back(std::thread(random_func, i, cmd_arg));
    }

    // Join threads
    for (std::thread& thrd : threads) {
        thrd.join();
    }

    std::cout << "All threads have finished finding numbers." << std::endl;
    return 0;
}