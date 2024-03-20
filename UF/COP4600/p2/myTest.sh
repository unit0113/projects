cd ./MemoryManager
make
cd ..
g++ -std=c++17 -o myTest myTest.cpp -L ./MemoryManager -lMemoryManager
#./myTest
valgrind --leak-check-full -s ./myTest