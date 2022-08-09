#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#include <string>

int main() {

    std::map<std::string, double> transactions;

    std::ifstream trans_file;

    std::string account, amount;
    double numeric_amount;

    trans_file.open("trans.dat");

    while (trans_file >> account >> amount) {
        numeric_amount = std::stod(amount);
    
        if (transactions.contains(account)) {
            transactions[account] += numeric_amount;
        } else {
            transactions[account] = numeric_amount;
        }
    }

    trans_file.close();


    std::ifstream old_mast_file;
    std::ofstream new_mast_file;

    std::string mast_account, name_first, name_last, balance;
    double numeric_balance;
    std::cout << std::fixed << std::setprecision(2);

    old_mast_file.open("oldmast.dat");
    new_mast_file.open("newmast.dat");

    while (old_mast_file >> mast_account >> name_first >> name_last >> balance) {
        numeric_balance = stod(balance);
        if (transactions.contains(mast_account)) {
            numeric_balance += transactions[mast_account];
            transactions.erase(mast_account);
        }
        new_mast_file << mast_account << ' ' << name_first << ' ' << name_last << ' ' << numeric_balance << std::endl;
    }

    old_mast_file.close();

    name_first = "None";
    name_last = "None";
    for (auto pair: transactions) {
        new_mast_file << pair.first << ' ' << name_first << ' ' << name_last << ' ' << pair.second << std::endl;
    }

    new_mast_file.close();
    
    return 0;
}