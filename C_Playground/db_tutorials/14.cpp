#include <iostream>
#include <vector>
#include <iterator>
using namespace std;

template<typename T>
void times2(T val) {
    cout << val << " * 2 = " << val * 2 << endl;
}

template<typename T>
T add(T v1, T v2) {
    return v1 + v2;
}

template<typename T>
T Max(T v1, T v2) {
    return (v1 > v2) ? v1 : v2;
}

template<typename T, typename U>
class Person {
    public:
        T height;
        U weight;
        static int numOfPeople;
        Person(T h, U w) {
            height = h;
            weight = w;
            numOfPeople++;
        }

        void getData() {
            cout << "Height: " << height << endl;
            cout << "Weight: " << weight << endl;
        }
};
template<typename T, typename U> int Person<T, U>::numOfPeople;         // To use static template member


int main() {
    times2(5);
    times2(10.4f);
    times2(3.14);

    cout << "5 + 4 = " << add(5, 4) << endl;
    cout << "5.5 + 1.4 = " << add(5.5f, 1.4f) << endl;
    cout << "Max of 4 and 8 = " << Max(4, 8) << endl;
    cout << "Max of cat and dog = " << Max("cat", "dog") << endl;

    Person<double, int> Mike(5.83, 284);
    Mike.getData();
    cout << "Number of People: " << Mike.numOfPeople << endl;


    vector<int> nums {1, 2, 3, 4};
    vector<int>::iterator itr;
    for (itr = nums.begin(); itr < nums.end(); itr++) {
        cout << *itr << endl;
    }

    vector<int>::iterator itr2 = nums.begin();
    advance(itr2, 2);
    cout << *itr2 << endl;

    auto itr3 = next(itr2, 1);
    cout << *itr3 << endl;

    auto itr4 = prev(itr2, 1);
    cout << *itr4 << endl;


    vector<int> nums2 {1, 4, 5, 6};
    vector<int> nums3 {2, 3};
    auto itr5 = nums2.begin();
    advance(itr5, 1);
    copy(nums3.begin(), nums3.end(), inserter(nums2, itr5));
    for (int& i: nums2) {
        cout << i << endl;
    }


}