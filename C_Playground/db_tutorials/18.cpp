#include <iostream>
#include <set>
#include <map>
#include <stack>
#include <queue>
#include <string>
using namespace std;

int main() {

    // Sets
    set<int> set1 {5, 4, 3, 2, 1, 1, 6, 7, 8};
    set1.insert(0);
    if (!set1.empty()) {
        for (const int& x: set1) {
            cout << x << '\n';
        }
    }

    set<int>::iterator it = set1.begin();
    advance(it, 1);
    cout << "Index 1: " << *it << '\n';
    it = set1.end();
    advance(it, -2);
    set1.erase(it, set1.end());
    cout << "After erase:\n";
    if (!set1.empty()) {
        for (const int& x: set1) {
            cout << x << '\n';
        }
    }

    int arr[] {6, 7, 8, 9, 10};
    set1.insert(arr, arr+5);
    auto val = set1.find(9);
    cout << "Found " << *val << endl;
    if (!set1.empty()) {
        for (const int& x: set1) {
            cout << x << '\n';
        }
    }

    auto eight = set1.lower_bound(8);   
    auto nine = set1.upper_bound(9);    

    cout << "\n*************************************************\n\n";

    multiset<int> mset1 {1, 1, 2, 4, 7, 3}; // allows duplicates
    if (!mset1.empty()) {
        for (const int& x: mset1) {
            cout << x << '\n';
        }
    }

    cout << "\n*************************************************\n\n";

    map<int, string> map1;
    map1.insert(pair<int, string>(1, "Geralt"));
    map1.insert(pair<int, string>(2, "Ciri"));
    map1.insert(pair<int, string>(3, "Triss"));
    map1.insert(pair<int, string>(4, "Yen"));

    map<int, string>::iterator it2;
    for (it2 = map1.begin(); it2 != map1.end(); it2++) {
        cout << "Key: " << it2->first << ", Value: " << it2->second << '\n';
    }

    //multimap allows duplicate keys with different values
    multimap<int, string> mmap1;
    mmap1.insert(pair<int, string>(1, "Geralt"));
    mmap1.insert(pair<int, string>(2, "Ciri"));
    mmap1.insert(pair<int, string>(1, "Triss"));
    mmap1.insert(pair<int, string>(4, "Yen"));

    multimap<int, string>::iterator it3;
    for (it3 = mmap1.begin(); it3 != mmap1.end(); it3++) {
        cout << "Key: " << it3->first << ", Value: " << it3->second << '\n';
    }

    cout << "\n*************************************************\n\n";

    //Container adapters
    stack<string> names;
    names.push("Bob");
    names.push("George");
    names.push("Jane");
    names.push("Charles");
    if (!names.empty()) {
        int size = names.size();
        for (int i{}; i < size; i++) {
            cout << names.top() << '\n';
            names.pop();
        }
    }

    cout << "\n*************************************************\n\n";

    queue<string> cast;
    cast.push("Fry");
    cast.push("Bender");
    cast.push("Leela");
    cast.push("Zoidberg");
    if (!cast.empty()) {
        int size = cast.size();
        for (int i{}; i < size; i++) {
            cout << cast.front() << '\n';
            cast.pop();
        }
    }

    cout << "\n*************************************************\n\n";

    priority_queue<int> nums;
    nums.push(4);
    nums.push(8);
    nums.push(5);
    nums.push(5);
    if (!nums.empty()) {
        int size = nums.size();
        for (int i{}; i < size; i++) {
            cout << nums.top() << '\n';
            nums.pop();
        }
    }

    cout << "\n*************************************************\n\n";

    //Enums
    enum day {Mon=1, Tues, Wed, Thur, Fri, Sat, Sun};
    enum day tuesday = Tues;
    cout << "Tuesday is the " << tuesday << "nd day of the week\n";


}