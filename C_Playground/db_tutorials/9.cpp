#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>


std::vector<int> gen_random_vec(int num_values, int min_val, int max_val);
std::vector<int> gen_fib_vec(int num_fib);

int main() {


    std::vector<int> vec_vals = gen_random_vec(10, 1, 50);

    for (auto val: vec_vals) {
        std::cout << val << std::endl;
    }

    std::sort(vec_vals.begin(), vec_vals.end(), [](int x, int y){return x < y;});       // [] is capture list, allows us to bring in outside values

    std::cout << "*********************************************" << std::endl;
    for (auto val: vec_vals) {
        std::cout << val << std::endl;
    }

    std::vector<int> even_vec_vals;
    std::copy_if(vec_vals.begin(), vec_vals.end(), std::back_inserter(even_vec_vals), [](int x){return (x % 2) == 0;});

    std::cout << "*********************************************" << std::endl;
    for (auto val: even_vec_vals) {
        std::cout << val << std::endl;
    }

    int sum{ 0 };
    std::for_each(vec_vals.begin(), vec_vals.end(), [&](int x){sum += x;});     // [&] equals acquire vals by reference
    std::cout << "Sum: " << sum << std::endl;

    std::cout << "*********************************************" << std::endl;
    int divisor;
    std::cout << "Divisor: ";
    std::cin >> divisor;
    std::vector<int> vec_div_vals;
    std::copy_if(vec_vals.begin(), vec_vals.end(), std::back_inserter(vec_div_vals), [divisor](int x){return (x % divisor) == 0;});

    for (auto val: vec_div_vals) {
        std::cout << val << std::endl;
    }

    std::cout << "*********************************************" << std::endl;
    std::vector<int> double_vec;
    std::for_each(vec_vals.begin(), vec_vals.end(), [&](int x){double_vec.push_back(x * 2);});

    for (auto val: double_vec) {
        std::cout << val << std::endl;
    }

    std::cout << "*********************************************" << std::endl;
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {6, 7, 8, 9, 10};
    std::vector<int> vec3(5);       // vec with length of 5
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), vec3.begin(), [](int x, int y){return x + y;});

    for (auto val: vec3) {
        std::cout << val << std::endl;
    }

    std::cout << "*********************************************" << std::endl;
    std::function<int(int)> fib = [&fib](int n){return n < 2 ? n : fib(n-1) + fib(n-2);};
    std::cout << "Fib of 5: " << fib(5) << std::endl;
    std::cout << "Fib of 10: " << fib(10) << std::endl;

    std::cout << "*********************************************" << std::endl;
    std::vector<int> fib_vec = gen_fib_vec(10);
    for (auto val: fib_vec) {
        std::cout << val << std::endl;
    }


}


std::vector<int> gen_random_vec(int num_values, int min_val, int max_val) {
    std::vector<int> vec_random;
    srand(time(NULL));
    int range = 1 + max_val - min_val;
    for (size_t i = 0; i < num_values; i++) {
        vec_random.push_back(rand() % range + min_val);
    }
    return vec_random;
}


std::vector<int> gen_fib_vec(int num_fib) {
    std::vector<int> return_vec;
    int i = 0;
    std::function<int(int)> fib = [&fib](int n){return n < 2 ? n : fib(n-1) + fib(n-2);};

    while (i < num_fib) {
        return_vec.push_back(fib(i++));
    }

    return return_vec;
}
