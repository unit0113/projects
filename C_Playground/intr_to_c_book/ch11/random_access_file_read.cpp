#include <fstream>
#include <iostream>
#include <string>

typedef struct {
    size_t account_number;
    char name_last[20];
    char name_first[20];
    double balance;
} Client_Data;

int main() {

    std::ifstream account_file("accounts.dat", std::ios::binary);

    Client_Data client = {0, "", "", 0.0};
    std::cout << "Enter Account Number: ";
    std::string account_string;
    std::cin >> account_string;
    client.account_number = stoi(account_string);

    std::string str_balance;
    while (client.account_number != 0) {
        account_file.seekg(client.account_number * sizeof(Client_Data));
        //account_file.read(reinterpret_cast<char*>(&client.account_number), sizeof(client.account_number));
        std::cout << client.account_number;
        account_file >> account_string >> client.name_last >> client.name_first >> str_balance;
        std::cout << account_string << ' ' << client.name_last << ' ' << client.name_first << ' ' << str_balance;
        client.balance = stod(str_balance);
        std::cout << "Account " << client.account_number << ": " << client.name_last << ", " << client.name_first << "; Balance: " << client.balance << std::endl;
    }






    return 0;
}