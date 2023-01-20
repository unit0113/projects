#include <string>
#include <iostream>
#include <forward_list>

using namespace std;

class Observer {
    public:
        Observer(string name): mName(name) {}

        void OnNotify() const {
            cout << mName << " says Hello There!\n";
        }

        bool operator==(const Observer& other) const {
            return other.mName == mName;
        }
    
    private:
        string mName;
};


class Subject {
    public:
        void AddObserver(const Observer& observer) {
            mObservers.push_front(observer);
        }

        void RemoveObserver(const Observer& observer) {
            mObservers.remove(observer);
        }

    void NotifyAll() {
        for (const auto& o: mObservers) {
            o.OnNotify();
        }
    }

    private:
        forward_list<Observer> mObservers;
};


int main() {
    Subject subject;
    Observer observer1("Observer-1");
    Observer observer2("Observer-2");
    Observer observer3("Observer-3");

    subject.AddObserver(observer1);
    subject.AddObserver(observer2);
    subject.AddObserver(observer3);

    subject.NotifyAll();

    subject.RemoveObserver(observer1);
    subject.NotifyAll();
}