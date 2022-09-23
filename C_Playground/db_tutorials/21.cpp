#include <iostream>
#include <string>
#include <regex>
#include <iterator>
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

    string str1 = "The cat cat fell out of the window";
    regex reg1 ("(\\b\\w+)\\s+\\1");        // \\1 = match group 1
    printMatches(str1, reg1);

    string str2 = "<a href='#'><b>The Link</b></a>";
    regex reg2 ("<b>(.*?)</b>");
    string result2;
    regex_replace(back_inserter(result2), str2.begin(), str2.end(), reg2, "$1");
    cout << result2 << endl;

    string str3 = "412-555-1212";
    regex reg3 ("(\\d{3})-(\\d{3}-\\d{4})");
    string result3;
    regex_replace(back_inserter(result3), str3.begin(), str3.end(), reg3, "($1) $2");
    cout << result3 << endl;

    string str4 = "http://www.youtube.com";
    regex reg4 ("https?://([\\w.]+)");
    string result4;
    regex_replace(back_inserter(result4), str4.begin(), str4.end(), reg4, "<a href='https://$1'>$1</a>\n");
    cout << result4 << endl;

    // Look aheads
    string str5 = " one two three four five ";
    regex reg5 ("(\\w+(?=\\b))");
    printMatches(str5, reg5);

    string str6 = "1. Dog 2. Cat 3. Turtle";
    regex reg6 ("\\d\\.\\s(Dog|Cat)");
    printMatches(str6, reg6);

    string str7 = "12345 12345-1234 1234 12346-333";
    regex reg7 ("(\\d{5}-\\d{4}|\\d{5}\\s)");
    printMatches(str7, reg7);

    string str8 = "14125551212 4125551212 (412)5551212 412 555 1212 412-555-1212 1-412-555-1212";
    regex reg8 ("((1?)(-| ?)(\\()?(\\d{3})(\\)|-| |\\)-|\\) )?(\\d{3})(-| )?(\\d{4}|\\d{4}))");
    printMatches(str8, reg8);


}