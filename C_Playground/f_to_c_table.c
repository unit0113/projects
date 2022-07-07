#include <stdio.h>

/*Print Fahrenheit->Celsius table 
for fahr = 0, 20, ..., 300 */

int main() {

    float fahr, celsius;
    int lower = 0;
    int upper = 300;
    int step = 20;

    printf("  F    C  \n");
    printf("----------\n");
    fahr = lower;
    while (fahr <= upper) {
        celsius = (5.0/9.0) * (fahr-32.0);
        printf("%3.0f %6.1f\n", fahr, celsius);
        fahr += step;
    }

    return 0;
}