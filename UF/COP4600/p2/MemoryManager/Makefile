default: MemoryManager

manager.o: MemoryManager.cpp MemoryManager.h
	 g++ -std=c++17 -c -o manager.o MemoryManager.cpp

MemoryManager: manager.o
	ar cr libMemoryManager.a manager.o

clean:
	-rm -f libMemoryManager.a
	-rm -f manager.o
