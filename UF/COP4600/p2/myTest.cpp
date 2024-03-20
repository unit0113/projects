#include "MemoryManager/MemoryManager.h"
#include <iostream>
#include <vector>
#include <inttypes.h>

void checkGetList(const MemoryManager& manager, const std::vector<uint16_t>& correctList);

int main() {
    unsigned wordSize = sizeof(uint64_t);
    size_t numberOfWords = 26;
    MemoryManager memoryManager(wordSize, bestFit);
    memoryManager.initialize(numberOfWords);
    std::vector<uint16_t> correctList = {1, 0, 26};

    // Test with no allocation
    std::cout << "No allocations" << std::endl;
    checkGetList(memoryManager, correctList);

    // First allocation
    /*uint64_t* testArray1 = static_cast<uint64_t*>(memoryManager.allocate(sizeof(uint64_t) * 10));
    correctList = {1, 10, 16};
    std::cout << "First allocation" << std::endl;
    checkGetList(memoryManager, correctList);

    uint64_t* testArray2 = static_cast<uint64_t*>(memoryManager.allocate(sizeof(uint64_t) * 2));
    correctList = {1, 12, 14};
    std::cout << "Second allocation" << std::endl;
    checkGetList(memoryManager, correctList);

    uint64_t* testArray3 = static_cast<uint64_t*>(memoryManager.allocate(sizeof(uint64_t) * 2));
    correctList = {1, 14, 12};
    std::cout << "Third allocation" << std::endl;
    checkGetList(memoryManager, correctList);
    
    
    uint64_t* testArray4 = static_cast<uint64_t*>(memoryManager.allocate(sizeof(uint64_t) * 6));
    correctList = {1, 20, 6};
    std::cout << "Fourth allocation" << std::endl;
    checkGetList(memoryManager, correctList);
    */

    memoryManager.shutdown();

    return 0;
}

void checkGetList(const MemoryManager& manager, const std::vector<uint16_t>& correctList) {
    std::cout << "Expected List:" << std::endl;
    for (const uint16_t& val : correctList) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    uint16_t* list = static_cast<uint16_t*>(manager.getList());
    if (!list) {
        std::cout << "No list received" << std::endl;
        std::cout << "FAILED" << std::endl;
        return;
    }

    std::cout << "Actual List:" << std::endl;
    bool correct = true;
    for (int i=0; i<2*list[0]+1; ++i) {
        std::cout << list[i] << ' ';
        if (list[i] != correctList[i]) {correct = false;}
    }
    std::cout << std::endl;

    free(list);
    if(correct) {std::cout << "PASSED" << std::endl;}
    else {std::cout << "FAILED" << std::endl;}
    std::cout << std::endl;
}