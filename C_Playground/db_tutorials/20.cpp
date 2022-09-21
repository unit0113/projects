#include <iostream>
#include <regex>
using namespace std;


void printMatches(string str, const regex& reg) {
    smatch matches;

    while (regex_search(str, matches, reg)) {
        cout << "Match: " << matches.str(1) << '\n';
        str = matches.suffix().str();
    }
    cout << endl;
}


int main() {

    string str1 = "cat cats dogs";
    regex reg1 ("([cat]+s?)");
    printMatches(str1, reg1);

    string str2 = "doctor doctors doctor's";
    regex reg2 ("([doctor]+['s]*)");
    printMatches(str2, reg2);

    string str3 = "Just some words\n"
        "and some more\r\n"
        "and more\n";
    regex reg3 ("(\r?\n)");
    string newStr3 = regex_replace(str3, reg3, " ");
    cout << newStr3 << endl;

    string str4 = "<name>Life on Mars</name>"
        "<name>Freaks and Geeks</name>";
    regex reg4 ("<name>(.*?)</name>");          // ? changes to lazy match
    printMatches(str4, reg4);

    string str5 = "ape at the apex";
    regex reg5 ("(\\bape\\b)");
    printMatches(str5, reg5);

    string str6 = "Match everything except the @";
    regex reg6 ("(^.*[^@])");
    printMatches(str6, reg6);

    string str7 = "@ Get this string";
    regex reg7 ("([^@\\s].*$)");
    printMatches(str7, reg7);

    string str8 = "206-709-3100 202-456-1111 212-832-2000 216-867-5309";
    regex reg8 ("(\\d{3}-\\d{4})");
    printMatches(str8, reg8);

    // multiple capture groups
    string str9 = "My number is 904-285-3700";
    regex reg9 ("(\\d{3})-(\\d{3})-(\\d{4})");
    smatch matches;
    if(regex_search(str9, matches, reg9)) {
        for (size_t i {1}; i < matches.size(); ++i) {
            cout << matches[i] << endl;
        }
    }




}