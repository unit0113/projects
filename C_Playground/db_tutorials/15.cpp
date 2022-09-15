#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;


class Pizza {
    public:
        virtual void MakePizza() = 0;
};

class NYStyleCrust {
    public:
        string AddIngredient() {
            return "Super thin crust\n"s;
        }
};

class DeepDishCrust {
    public:
        string AddIngredient() {
            return "Awesome Chicago style crust\n"s;
        }
};

template <typename T>
class MeatLovers: public T {
    public:
        string AddIngredient() {
            return "Lots o' meat, "s + T::AddIngredient();
        }
};

template <typename T>
class Supreme: public T {
    public:
        string AddIngredient() {
            return "Meat and veggies, "s + T::AddIngredient();
        }
};

template <typename T>
class MeatNY: public T, public Pizza {
    public:
        void MakePizza() override {
            cout << "NY style meat lovers pizza: " << T::AddIngredient();
        }
};

template <typename T>
class DeepDishSupreme: public T, public Pizza {
    public:
        void MakePizza() override {
            cout << "Deep dish supreme pizza: " << T::AddIngredient();
        }
};


int main() {

    int size;
    cout << "Enter number of ints to store: ";
    cin >> size;

    unique_ptr<int[]> nums = make_unique<int[]>(size);

    if (nums != NULL) {
        for (size_t i {}; i < size; i++) {
            cout << "Enter a number: ";
            cin >> nums[i];
        }
    }

    cout << "Entered numbers:\n";
    for (size_t i {}; i < size; i++) {
        cout << nums[i] << endl;
    }



    vector<unique_ptr<Pizza>> pizzaOrders;
    pizzaOrders.emplace_back(make_unique<MeatNY<MeatLovers<NYStyleCrust>>>());
    pizzaOrders.emplace_back(make_unique<DeepDishSupreme<Supreme<DeepDishCrust>>>());

    for (const auto& pizza: pizzaOrders) {
        pizza->MakePizza();
    }



}