#include "MemoryManager.h"
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <utility>
#include <algorithm>
#include <fcntl.h>
#include <string>
#include <unistd.h>

//-------- Allocators ----------//
int bestFit(int sizeInWords, void *list) {
    // Check input
    if (list == nullptr) {return -1;}

    // Cast list
    uint16_t* castList = static_cast<uint16_t*>(list);
    uint16_t size = castList[0];

    // Loop vars
    int fit;
    int topFit = std::numeric_limits<int>::max();
    int bestIndex = -1;

    // Iterate through new list
    for (int i=1; i<size*2+1; i+=2) {
        fit = castList[i+1] - sizeInWords;
        if (fit >= 0) {
            if (fit < topFit) {
                topFit = fit;
                bestIndex = i;
            }
        }
    }

    if (bestIndex == -1) {return -1;}
    return castList[bestIndex];
}

int worstFit(int sizeInWords, void *list) {
    // Check input
    if (list == nullptr) {return -1;}

    // Cast list
    uint16_t* castList = static_cast<uint16_t*>(list);
    uint16_t size = castList[0];

    // Loop vars
    int fit;
    int topFit = -1;
    int bestIndex = -1;

    // Iterate through new list
    for (int i=1; i<size*2+1; i+=2) {
        fit = castList[i+1] - sizeInWords;
        if (fit >= 0) {
            if (fit > topFit) {
                topFit = fit;
                bestIndex = i;
            }
        }
    }

    if (bestIndex == -1) {return -1;}
    return castList[bestIndex];
}

//-------- Memory Manager ----------//
MemoryManager::MemoryManager(unsigned int wordSize, std::function<int(int, void *)> allocator) : wordSize(wordSize), allocator(allocator), base(nullptr) {}

MemoryManager::~MemoryManager() {
    shutdown();
}

void MemoryManager::initialize(size_t newSizeInWords) {
    if (base) {shutdown();}
    if (newSizeInWords > 65536) {std::invalid_argument("Word limit is 65536");}
    sizeInWords = newSizeInWords;

    // Get array of bytes, store start address in base
    //sbrk: https://www.man7.org/linux/man-pages/man2/brk.2.html
    base = sbrk(sizeInWords * wordSize);
    //base = new char[sizeInWords * wordSize];

    // Initialize holes
    holes[0] = new Block(0, sizeInWords);
}

void MemoryManager::shutdown() {
    if (allocated.size() > 0) {
        for (std::pair<unsigned int, Block*> kv : allocated) {
            delete kv.second;
        }
    }
    allocated.clear();

    if (holes.size() > 0) {
        for (std::pair<unsigned int, Block*> kv : holes) {
            delete kv.second;
        }
    }
    holes.clear();

    //if (base) {delete[] base;}
    if (base) {brk(base);}
    base = nullptr;
}

void *MemoryManager::allocate(size_t sizeInBytes) {
    int numWords = sizeInBytes / wordSize;
    // Find block
    uint16_t* list = static_cast<uint16_t*>(getList());
    int offset = allocator(numWords, list);
    delete[] list;

    // Check if valid block was found
    if (offset == -1) {return nullptr;}

    // Add to allocated
    allocated[offset] = new Block(offset, numWords);

    // Adjust holes
    if (holes[offset]->size() != numWords) {
        holes[offset+numWords] = new Block(offset+numWords, holes[offset]->size() - numWords);
    }
    delete holes[offset];
    holes.erase(offset);
    //return base + offset * wordSize;
    return static_cast<char*>(base) + offset * wordSize;
}

void MemoryManager::free(void *address) {
    //Remove block from allocated
    //Block* mergeBlock = allocated[(static_cast<char*>(address)-base)/wordSize];
    Block* mergeBlock = allocated[(static_cast<char*>(address)-static_cast<char*>(base))/wordSize];
    allocated.erase(mergeBlock->begin());

    // Defrag holes
    bool mustAdd = true;
    for (std::pair<unsigned int, Block*> hole : holes) {
        // Check forwards if block can be merged
        if (hole.second->end() == mergeBlock->begin()) {
            hole.second->merge(mergeBlock);
            delete mergeBlock;
            mergeBlock = hole.second;
            mustAdd = false;
        }
        // Check backwards if block can be merged
        if (hole.second->begin() == mergeBlock->end()) {
            mergeBlock->merge(hole.second);
            delete hole.second;
            holes.erase(hole.first);
            break;
        }
        else if (hole.second->begin() > mergeBlock->end()) {break;}
    }

    if (mustAdd) {holes[mergeBlock->begin()] = mergeBlock;}
}

void MemoryManager::setAllocator(std::function<int(int, void *)> newAllocator) {
    allocator = newAllocator;
}

int MemoryManager::dumpMemoryMap(char *filename) {
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0777);

    // Fail if file did not open successfully
    if (fd == -1) {return -1;}

    // Assemble string
    std::string holeStr = "";
    for (std::pair<unsigned int, Block*> hole : holes) {
        if (holeStr.size() > 0) {holeStr.append(" - ");}
        holeStr.append("[");
        holeStr.append(std::to_string(hole.second->begin()));
        holeStr.append(", ");
        holeStr.append(std::to_string(hole.second->size()));
        holeStr.append("]");
    }

    // Write to file
    size_t writeSize = write(fd, holeStr.data(), holeStr.size());
    
    // Confirm successful write
    if (writeSize != holeStr.size()) {return -1;}

    close(fd);
    return 0;    
}

void *MemoryManager::getList() const {
    // Check if memory has been allocated
    if (base == nullptr) {return nullptr;}

    // Check if available memory
    if (holes.size() == 0) {return nullptr;}

    // Initialize array
    uint16_t* castList = new uint16_t[holes.size()*2 + 1];
    castList[0] = holes.size();

    // Iterate through holes
    int i = 1;
    for (const std::pair<unsigned int, Block*> hole : holes) {
        castList[i] = hole.first;
        castList[i+1] = hole.second->size();
        i += 2;
    }

    // Cast to void
    void* list = static_cast<void*>(castList);
    return list;
}

void *MemoryManager::getBitmap() const {
    // Initialize bitmap
    uint16_t arrLength = sizeInWords / 8;
    if (sizeInWords % 8 != 0) {++arrLength;}
    uint8_t *bitMap = new uint8_t[2 + arrLength];
    std::fill_n(bitMap+2, arrLength, 0xFF);

    // Add hole data
    for (const std::pair<unsigned int, Block*> hole : holes) {
        for (int i = hole.second->begin(); i < hole.second->end(); ++i) {
            clearBit(bitMap, i+16);
        }
    }

    // Trailing 0's
    for (int i = sizeInWords; i < arrLength * 8; ++i) {
        clearBit(bitMap, i+16);
    }

    // Add size data
    bitMap[0] = arrLength & 0xFF;
    bitMap[1] = arrLength >> 8;

    return bitMap;
}

unsigned int MemoryManager::getWordSize() const {
    return wordSize;
}

void *MemoryManager::getMemoryStart() const {
    return base;
}

unsigned int MemoryManager::getMemoryLimit() const {
    return wordSize * sizeInWords;
}