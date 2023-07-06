#include <vector>
#include <iostream>

using namespace std;

class unorderedSet {
private:
    int m_size{};
    int m_maxSize{2};
    float m_maxLoadFactor = 0.6f;
    vector<int> vec;
    int hash(int val) {return (val * 31) % m_maxSize;}
    float loadFactor() {return m_size / (float)m_maxSize;}
    void resize() {
        m_maxSize *= 2;
        vector<int> newVec(m_maxSize, 0);

        int index{};
        for (const int& x : vec) {
            if (x != 0) {
                index = hash(x);
                while (newVec[index] != 0) {
                    index = (index + 1) % m_maxSize;
                }
                newVec[index] = x;
            }
        }
        vec = newVec;
    }
    
public:
    unorderedSet() : vec(m_maxSize, 0) {}

    void insert(int val) {
        if (has(val)) {return;}

        int index = hash(val);
        while (vec[index] != 0) {
            index = (index + 1) % m_maxSize;
        }
        vec[index] = val;
        ++m_size;
        
        if (loadFactor() > m_maxLoadFactor) {resize();}
    }

    bool has(int val) {
        int index = hash(val);
        while (vec[index] != 0) {
            if (vec[index] == val) {return true;}
            index = (index + 1) % m_maxSize;
        }
        return false;
    }

    void rem(int val) {
        int index = hash(val);
        while (vec[index] != 0) {
            if (vec[index] == val) {
                vec[index] = 0;
                --m_size;
                return;
            }
            index = (index + 1) % m_maxSize;
        }
    }

    int size() {
        return m_size;
    }
};


int main(){
	unorderedSet s;
    s.insert(1);
    cout << s.size() << endl;
    cout << boolalpha << s.has(1) << endl;
    s.insert(1);
    cout << s.size() << endl;
    cout << boolalpha << s.has(1) << endl;
    s.rem(1);
    cout << s.size() << endl;
    cout << boolalpha << s.has(1) << endl;

    for (int x = 1; x <= 100; ++x) {
        s.insert(x);
    }
    cout << s.size() << endl;
}