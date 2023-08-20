// write a program that takes in 3 variables and performs three math functions. The results MUST be printed in the functions.
// Geneva Anderson, Nov 2021

#include <stdio.h>

int doubleIt(int x)
{
    x = x * 2;
    printf("%d\n", x);
    return x;
}

int addThem(int x, int y)
{
    int val = x + y;
    printf("%d, %d\n", x, y);
    return val;
}

int halfIt(int x)
{
    x = x / 2;
    printf("%d\n", x);
    return x;
}

int main()
{
    int x = 2;
    int y = 4;
    int first = doubleIt(x);
    int second = addThem(x, y);
    int third = halfIt(x);

    return 0;
}