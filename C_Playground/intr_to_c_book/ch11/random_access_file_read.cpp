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
        account_file.seekg(client.account_number * sizeof(Client_Data), std::ios::beg);
        account_file.read((char *) &client.account_number, sizeof(client.account_number));
        std::cout << client.account_number << std::endl;
        account_file.read((char *) &client.name_last[0], sizeof(client.name_last));
        account_file.read((char *) &client.name_first[0], sizeof(client.name_first));
        account_file.read((char *) &client.balance, sizeof(client.balance));
        
        std::cout << "Account " << client.account_number << ": " << client.name_last << ", " << client.name_first << "; Balance: " << client.balance << std::endl;

        std::cout << "Enter Account Number: ";
        std::string account_string;
        std::cin >> account_string;
        client.account_number = stoi(account_string);
    }

    return 0;
}