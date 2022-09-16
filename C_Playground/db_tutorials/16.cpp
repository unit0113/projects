#include <iostream>
#include <ctime>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <cmath>
using namespace std;

int getRand(const int max) {
    return rand() % (max + 1);
}

void executeThread(int id) {
    auto now = chrono::system_clock::now();
    time_t sleepTime = chrono::system_clock::to_time_t(now);
    tm local = *localtime(&sleepTime);

    cout << "Thread " << id << " Sleep time: " << ctime(&sleepTime) << endl;

    this_thread::sleep_for(chrono::seconds(getRand(3)));
    now = chrono::system_clock::now();
    sleepTime = chrono::system_clock::to_time_t(now);
    cout << "Thread " << id << " Awake time: " << ctime(&sleepTime) << endl;
}

string getTime() {
    auto now = chrono::system_clock::now();
    time_t sleepTime = chrono::system_clock::to_time_t(now);
    return ctime(&sleepTime);
}

double ACCTBALANCE = 100;

mutex acctLock;

void getMoney(int id, double amount) {
    lock_guard<mutex> lock(acctLock);
    this_thread::sleep_for(chrono::seconds(3));
    cout << id << " tries to withdraw $" << amount << " at " << getTime() << endl;
    if ((ACCTBALANCE - amount) >= 0) {
        ACCTBALANCE -= amount;
        cout << "New balance: $" << ACCTBALANCE << endl;
    } else {
        cout << "Insufficient funds\nCurrent balance is $" << ACCTBALANCE << endl;
    }
}

void findPrimes(unsigned int start, unsigned int end, vector<unsigned int>& vec) {
    if (start % 2 == 0) {
        start++;
    }

    for (size_t i = start; i <= end; i += 2) {
        for (size_t j = 2; j < i; j++) {
            if (i % j == 0) {
                break;
            } else if (j + 1 == i) {
                vec.push_back(i);
            }
        }
    }
}

mutex vecLock;
vector<unsigned int> primeVec;

void findPrimesThreaded(unsigned int start, unsigned int end) {
    if (start % 2 == 0) {
        start++;
    }

    for (size_t i = start; i <= end; i += 2) {
        for (size_t j = 2; j < i; j++) {
            if (i % j == 0) {
                break;
            } else if (j + 1 == i) {
                vecLock.lock();
                primeVec.push_back(i);
                vecLock.unlock();
            }
        }
    }
}

void primesThreadManager(unsigned int start, unsigned int end, unsigned int numThreads) {
    vector<thread> threadVec;
    unsigned int threadSpread = end / numThreads;
    unsigned int newEnd = start + threadSpread - 1;

    for (size_t i {}; i < numThreads; i++) {
        threadVec.emplace_back(thread(findPrimesThreaded, start, newEnd));
        start += threadSpread;
        newEnd += threadSpread;
    }

    for (auto& t: threadVec) {
        t.join();
    }
}

int main() {
    srand(time(NULL));

    /*    
    thread t1(executeThread, 1);
    t1.join();
    thread t2(executeThread, 2);
    t2.join();
    thread t3(executeThread, 3);
    t3.join();
    */

    /*
    thread threads[10];
    for (size_t i{}; i < 10; i++) {
        threads[i] = thread(getMoney, i, 15);
    }

    for (size_t i{}; i < 10; i++) {
        threads[i].join();
    }
    */


    /*
    vector<unsigned int> vec;
    int start = clock();
    findPrimes(1, 100'000, vec);
    for (const auto& x: vec) {
        cout << x << '\n';
    }
    int end = clock();
    cout << "Time: " << (end - start) / double(CLOCKS_PER_SEC) << endl;
    */

    int start = clock();
    primesThreadManager(1, 100'000, 4);
    int end = clock();

    for (const auto& i: primeVec) {
        cout << i << '\n';
    }

    cout << "Time: " << (end - start) / double(CLOCKS_PER_SEC) << endl;

}