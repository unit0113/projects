#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "ABS.h"
#include "ABQ.h"
using namespace std;

int main() {
    vector<double> scaleVec {1.5, 2.0, 3.0, 10.0, 50.0, 100.0};
    vector<int> nVec {10'000, 30'000, 50'000, 75'000, 100'000, 150'000};

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> duration;
    ABS<int> *stack;
    ABQ<int> *queue;
    int resizes;

    for (const auto& scale: scaleVec) {
        for (const auto& n: nVec) {
            // Test stack push
            stack = new ABS<int>(2, scale);
            start = std::chrono::system_clock::now();
            for (int i{}; i < n; ++i) {
                stack->push(i);
            }

            end = std::chrono::system_clock::now();
            duration = end - start;
            resizes = stack->getTotalResizes();
            cout << "Stack-Push SF: " << left << setw(3) << scale << " N: " << setw(6) << n << " Resizes: " << resizes << " Duration: " << duration << endl;
            
            // Test stack pop
            start = std::chrono::system_clock::now();
            for (int i{}; i < n; ++i) {
                stack->pop();
            }
            end = std::chrono::system_clock::now();
            duration = end - start;
            cout << "Stack-Pop SF: " << left << setw(3) << scale << " N: " << setw(6) << n << " Resizes: " << stack->getTotalResizes() - resizes << " Duration: " << duration << endl;
            delete stack;

            // Test queue enqueue
            queue = new ABQ<int>(2, scale);
            start = std::chrono::system_clock::now();
            for (int i{}; i < n; ++i) {
                queue->enqueue(i);
            }
            end = std::chrono::system_clock::now();
            duration = end - start;
            resizes = queue->getTotalResizes();
            cout << "Queue-Enqueue SF: " << left << setw(3) << scale << " N: " << setw(6) << n << " Resizes: " << queue->getTotalResizes() << " Duration: " << duration << endl;

            // Test queue dequeue
            start = std::chrono::system_clock::now();
            for (int i{}; i < n; ++i) {
                queue->dequeue();
            }
            end = std::chrono::system_clock::now();
            duration = end - start;
            cout << "Queue-Dequeue SF: " << left << setw(3) << scale << " N: " << setw(6) << n << " Resizes: " << queue->getTotalResizes() - resizes << " Duration: " << duration << endl;
            delete queue;
        }
    }




    //start = std::chrono::system_clock::now();
    //end = std::chrono::system_clock::now();
    //std::chrono::duration<double> elapsed_seconds = end - start;



}