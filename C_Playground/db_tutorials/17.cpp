#include <iostream>
#include <deque>
#include <list>
#include <forward_list>

using namespace std;


bool isEven(const int val) {
    return (val % 2) ==0;
}

int main() {

    deque<int> deq1;
    deq1.push_back(5);
    deq1.push_front(1);
    deq1.assign({11, 12});      // overwrites
    deq1.push_back(5);
    deq1.push_front(1);
    cout << "Size: " << deq1.size() << '\n';
    cout << "deq1[0]: " << deq1[0] << '\n';
    cout << "deq1.at(0): " << deq1.at(0) << '\n';
    deque<int>::iterator it = deq1.begin() + 1;
    deq1.insert(it, 3);
    int tempArr[5] = {6, 7, 8, 9, 10};
    deq1.insert(deq1.end(), tempArr, tempArr + 5);

    for (const auto& x: deq1) {
        cout << x << '\n';
    }

    cout << "After erases: \n";
    deq1.erase(deq1.end());
    deq1.erase(deq1.begin(), deq1.begin() + 2);
    deq1.pop_front();
    deq1.pop_back();

    for (const auto& x: deq1) {
        cout << x << '\n';
    }

    cout << "\n*********************************\n\n";
    int arr[5] = {1, 2, 3, 4, 5};
    list<int> list1;
    list1.insert(list1.begin(), arr, arr + 5);
    list1.assign({10, 20, 30});                     //erases everything else
    list1.push_back(6);
    list1.push_front(0);
    cout << "Size: " << list1.size() << "\n"; 
    list<int>::iterator it2 = list1.begin();
    advance(it2, 2);
    cout << "List index 2: " << *it2 << '\n';
    it2 = list1.begin();
    list1.insert(it2, 8);


    for (const auto x: list1) {
        cout << x << '\n';
    }

    cout << "Erasing elements: \n";
    it2 = list1.begin();
    advance(it2, 3);
    list1.erase(it2);

    for (const auto x: list1) {
        cout << x << '\n';
    }

    cout << "After popping: \n";
    list1.pop_front();
    list1.pop_back();

    for (const auto x: list1) {
        cout << x << '\n';
    }

    cout << "\n*********************************\n\n";

    int arr2[6] = {10, 9, 8, 7, 6, 6};
    list<int> list2;
    list1.insert(list2.begin(), arr2, arr2 + 6);

    cout << "Pre Sort: \n";
    for (const auto x: list2) {
        cout << x << '\n';
    }
    cout << "Post Sort: \n";
    list2.sort();
    for (const auto x: list2) {
        cout << x << '\n';
    }

    cout << "Sorted unique values reversed: \n";
    list2.reverse();
    list2.unique();
    for (const auto x: list2) {
        cout << x << '\n';
    }

    cout << "Evens removed: \n";
    list2.remove_if(isEven);
    for (const auto x: list2) {
        cout << x << '\n';
    }

    list1.merge(list2);
    cout << "Merged list: \n";
    for (const auto x: list1) {
        cout << x << '\n';
    }

    //Forward list is a one directional linked list

}