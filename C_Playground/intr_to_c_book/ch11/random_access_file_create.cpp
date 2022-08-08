#include <fstream>
#include <iostream>
#include <string>

typedef struct {
    size_t account_number;
    char name_last[20];
    char name_first[20];
    double balance;
} Client_Data;

int main () {

    std::ofstream account_file("accounts.dat", std::ios::binary);

    Client_Data client = {0, "", "", 0.0};
    std::cout << "Enter Account Number: ";
    std::string account_string;
    std::cin >> account_string;
    client.account_number = stoi(account_string);

    std::string str_balance;
    while (client.account_number != 0) {
        std::cout << "Enter Last Name First Name and Current Account Balance: ";
        std::cin >> client.name_last >> client.name_first >> str_balance;
        client.balance = stod(str_balance);

        // Move to location and input data
        account_file.seekp(client.account_number * sizeof(Client_Data), std::ios::beg);
        account_file.write((char *) &client.account_number, sizeof(client.account_number));
        account_file.write((char *) &client.name_last[0], sizeof(client.name_last));
        account_file.write((char *) &client.name_first[0], sizeof(client.name_first));
        account_file.write((char *) &client.balance, sizeof(client.balance));

        // Ask for new data
        std::cout << "Enter Account Number: ";
        std::cin >> account_string;
        client.account_number = stoi(account_string);
    }
    
    account_file.close();

    return 0;
}