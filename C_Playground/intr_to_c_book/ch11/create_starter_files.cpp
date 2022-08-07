#include <fstream>
#include <iostream>

int main() {

    std::ofstream mast_file;
    mast_file.open("oldmast.dat");

    mast_file << "100 Alan Jones 348.17" << std::endl;
    mast_file << "300 Mary Smith 27.19" << std::endl;
    mast_file << "500 Sam Sharp 0.00" << std::endl;
    mast_file << "700 Suzy Green -14.22" << std::endl;

    mast_file.close();


    std::ofstream trans_file;
    trans_file.open("trans.dat");

    trans_file << "100 27.14" << std::endl;
    trans_file << "300 62.11" << std::endl;
    trans_file << "400 100.56" << std::endl;
    trans_file << "900 82.17" << std::endl;

    trans_file.close();

    return 0;
}