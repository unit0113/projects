#pragma once
#include <functional>
#include <map>
#include <unordered_map>

//-------- Allocators ----------//
int bestFit(int sizeInWords, void *list);
int worstFit(int sizeInWords, void *list);


//-------- Block ----------//
struct Block {
    unsigned int wordOffset;
    size_t sizeInWords;

    Block(unsigned int wordOffset, size_t sizeInWords) : wordOffset(wordOffset), sizeInWords(sizeInWords) {};
    unsigned int begin() const {return wordOffset;};
    unsigned int end() const {return wordOffset + sizeInWords;};
    size_t size() const {return sizeInWords;};
    void merge(const Block* other) {sizeInWords += other->size();};
};

//-------- Memory Manager ----------//
class MemoryManager {
    private:
    unsigned int wordSize;
    std::function<int(int, void *)> allocator;
    size_t sizeInWords;
    void *base;
    std::unordered_map<unsigned int, Block*> allocated;
    std::map<unsigned int, Block*> holes;

    // https://stackoverflow.com/questions/2525310/how-to-define-and-work-with-an-array-of-bits-in-c
    // Note: this seems to mirror as well?
    void clearBit(uint8_t arr[], int index) const {arr[index / 8] &= ~(1 << (index % 8));};

    public:
    MemoryManager(unsigned int wordSize, std::function<int(int, void *)> allocator);
    ~MemoryManager();
    void initialize(size_t sizeInWords);
    void shutdown();
    void *allocate(size_t sizeInBytes);
    void free(void *address);
    void setAllocator(std::function<int(int, void *)> allocator);
    int dumpMemoryMap(char *filename);
    void *getList() const;
    void *getBitmap() const;
    unsigned int getWordSize() const;
    void *getMemoryStart() const;
    unsigned int getMemoryLimit() const;

};