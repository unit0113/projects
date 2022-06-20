#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <stdbool.h>


int main() {

    // ints
    int a = 10;
    int b;

    printf("Int max val: %d\n", INT_MAX);
    printf("Int min val: %d\n", INT_MIN);

    short c = 500;

    printf("Short max val: %d\n", SHRT_MAX);
    printf("Short min val: %d\n", SHRT_MIN);

    long d = 5000000;
    long long e = 5;

    printf("%ld\n", d);
    printf("%lld\n", e);
    printf("Int size: %d\n", sizeof(d));

    //Long same as int in windows?
    //Note %ld and %lld in prints
    printf("Long max val: %ld\n", LONG_MAX);
    printf("Long min val: %ld\n", LONG_MIN);
    printf("Long long max val: %lld\n", LLONG_MAX);
    printf("Long long min val: %lld\n", LLONG_MIN);

    unsigned int f = 20000000;
    //%u for unsigned
    //%lu for unsigned long
    printf("%u\n", f);
    printf("Unsigned max val: %u\n", UINT_MAX);

    //floats

    // 6 significant digits of precision
    float f1 = 12.123456789;
    printf("%.10f\n", f1);
    printf("Float size: %d\n", sizeof(f1));

    double d1 = 12.123456789101112131415;
    printf("Max float: %f\n", FLT_MAX);
    printf("Max double: %f\n", DBL_MAX);

    // Chars
    char c1 = 'A';
    char c2 = 90;

    // print ascii
    printf("%d\n", c1);
    printf("%d\n", c2);

    // print chars
    printf("%c\n", c1);
    printf("%c\n", c2);

    //bools
    bool z = true;


    return 0;
}