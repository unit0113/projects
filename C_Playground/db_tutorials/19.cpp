#include <iostream>
#include <regex>
#include <string>
using namespace std;


void printMatches(string str, const regex& reg) {
    smatch matches;
    cout << boolalpha;
    while (regex_search(str, matches, reg)) {
        cout << "Is there a match: " << matches.ready() << endl;
        cout << "Are there no matches: " << matches.empty() << endl;
        cout << "Number of matches: " << matches.size() << endl;
        cout << "Match: " << matches.str(1) << endl;
        str = matches.suffix().str();
        cout << endl;
    }
}


void printMatches2(string str, const regex& reg) {
    sregex_iterator currMatch(str.begin(), str.end(), reg);
    sregex_iterator lastMatch;
    while (currMatch != lastMatch) {
        smatch match = *currMatch;
        cout << match.str() << endl;
        currMatch++;
    }
    cout << endl;
}


int main() {
    string str = "The ape was at the apex";
    smatch matches;
    regex reg("(ape[^ ]?)");

    printMatches(str, reg);

    string str2 = "I picked a pickle";
    regex reg2("(pick([^ ]+)?)");
    printMatches2(str2, reg2);

    string str3 = "Cat rat mat fat pat papa";
    regex reg3("([crmfp]at)");      //list of options
    printMatches2(str3, reg3);
    regex reg4("([C-Fc-f]at)");     //ranges
    printMatches2(str3, reg4);
    regex reg5("([^Cr]at)");        //Not
    printMatches2(str3, reg5);

    //Replace
    regex reg6("([Cr]at)");
    string owlFood = regex_replace(str3, reg6, "Owl");
    cout << owlFood << endl;

    string str4 = "F.B.I. I.R.S. CIA";
    regex reg7("([^ ]\\..\\..\\.)");
    printMatches2(str4, reg7);

    string str5 = "This is a\nMultiline string\n"
        "That has many lines";
    regex reg8("\n");
    string noBreaks = regex_replace(str5, reg8, " ");
    cout << noBreaks << endl;

    // \d = [0-9]
    // \D = [^0-9]

    string str6 = "123 12345 123456 1234567 123456789 ";
    regex reg9 ("\\d{5,7}");     // 5-7 matches of \dat
    printMatches2(str6, reg9);


    // \w = [a-ZA-Z0-9]
    // \W = !\w

    string str11 = "412-867-5309";
    regex reg11 ("\\w{3}-\\w{3}-\\w{4}");
    printMatches2(str11, reg11);

    // \s = whitespace
    // \S = ! \s
    
    string str12 = "Isaruko Yamammoto";
    regex reg12 ("\\w{2,20}\\s\\w{2,20}");
    printMatches2(str12, reg12);
}