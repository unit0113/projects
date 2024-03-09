#include <iostream>
#include <string>
#include <fstream>

int main() {
    std::string line;
    std::ifstream pipeFile{"ex7_Pipe"};
    size_t line_num = 0;
    while(std::getline(pipeFile, line)) {
        ++line_num;
    }
    std::cout << "Program failed on operation " << line_num << std::endl;
    return 0;
}