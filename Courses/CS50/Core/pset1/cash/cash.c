#include <stdio.h>
#include <cs50.h>
#include <math.h>

int main(void)
{
    printf("Welcome to the super awesome change calculator 1.0!\n");
    float change;
    int q = 0;
    int d = 0;
    int n = 0;
    int p = 0;
    int coins = 0;

    //Get float for change owed, postive numbers only
    do
    {
        change = get_float("How much change is due?\n");
    }
    while (change < 0);

    //Change into cents
    int cents = round(change * 100);

    //number of quarters
    while (cents > 24)
    {
        q = q + 1;
        cents = cents - 25;
    }
    //printf("Quarters: %i\n", q);

    //number of dimes
    while (cents > 9)
    {
        d = d + 1;
        cents = cents - 10;
    }
    //printf("Dimes: %i\n", d);

    //number of nickles
    while (cents > 4)
    {
        n = n + 1;
        cents = cents - 5;
    }
    //printf("Nickles: %i\n", n);

    //number of pennies
    while (cents > 0)
    {
        p = p + 1;
        cents = cents - 1;
    }
    //printf("Pennies: %i\n", p);

    //Total number of coins
    coins = q + d + n + p;
    printf("Minimum coins: %i\n", coins);

}