cd ./MemoryManager
make
cd ..
g++ -std=c++17 -o CommandLineTest CommandLineTest.cpp -L ./MemoryManager -lMemoryManager
valgrind --leak-check-full -s ./CommandLineTest