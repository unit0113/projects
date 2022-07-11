#include <iostream>

void func1() {
    std::cout << "Func 1" << std::endl;
}

void func2() {
    std::cout << "Func 2" << std::endl;
}

int main() {
    int a { 10 };
    std::cout << &a << std::endl;

    int *mypointer = &a;
    std::cout << mypointer << std::endl;
    std::cout << *mypointer << std::endl;       // dereference, data at address

    *mypointer = 20;        // change data at location
    std::cout << a << std::endl;    // changes a because same memory

    int arr[100];
    std::cout << &arr << std::endl;
    std::cout << &arr[0] << std::endl;      // same as above
    std::cout << &arr[1] << std::endl;      // 4 bytes later

    int *arrayptr = arr;
    std::cout << arrayptr << std::endl;

    // pointer math
    int *ptr = &a;
    std::cout << ptr << std::endl;
    std::cout << ++ptr << std::endl;
    std::cout << ++ptr << std::endl;

    int arr2[10];
    int *first_index = arr2;
    // initilize arr values
    for(int i = 0; i < 10; i++){
        *(first_index + i) = i * 20;
    }

    for (int x : arr2) {
        std::cout << x << std::endl;
    }


    // best practices
    //when done with ptrs
    ptr = nullptr;


    // void pointers, can point to anything
    int d = 20;
    void *vp = &d;
    std::cout << vp << std::endl;
    std::cout << *(int*)vp << std::endl;       // must cast to object type before dereferencing

    // char ptrs
    char mychar = 'A';
    std::cout << (void *) &mychar << std::endl; //must cast to void ptrs to get address of mychar, otherwise returns the value as a str

    // ptr ptrs
    int e = 10;
    int *ptr_e = &e;
    int **pptr_e = &ptr_e;
    std::cout << ptr_e << std::endl;
    std::cout << pptr_e << std::endl;
    std::cout << *pptr_e << std::endl;

    // func ptrs
    void(*funcptr)();
    funcptr = func1;
    funcptr();
    funcptr = func2;
    funcptr();


    return 0;
}