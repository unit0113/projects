#include <stdio.h>
#include <cs50.h>
#include <math.h>

int main(void)
{
    long runsum = 0;
    int checksum = 0;
    int sum = 0;
    int sum1 = 0;
    long div = 10;
    long rem = 0;
    int remv = 0;
    long cc = 0;
    printf("Welcome to the super awesome credit card number combobulator 1.0!\n");
    do
    {
        cc = get_long("Card Number?\n");

    }
    while (cc < 1);

    //Check if length = 16
    if (cc > 999999999999999 && cc <= 9999999999999999)
    {
        for (int i = 16; i > 0; i--)
        {
            //For odd digits
            if (i % 2 == 0)
            {
                rem = cc % div;
                rem = rem - runsum;
                runsum = runsum + rem;
                rem = rem / (div / 10);
                div = div * 10;
                checksum = checksum + rem;

                //Prog Checks
                //printf("Digit: %i\n", i);
                //printf("%lu\n", runsum);
                //printf("%i\n", checksum);
            }

            //For even digits
            else if (i % 2 == 1)
            {
                rem = cc % div;
                rem = rem - runsum;
                runsum = runsum + rem;
                rem = rem / (div / 10);
                div = div * 10;

                //Check if rem * 2 is 10 or greater
                if (rem * 2 > 9)
                {
                    rem = rem * 2;
                    sum = rem % 10;
                    checksum = checksum + sum;
                    sum1 = rem % 100;
                    sum1 = sum1 - sum;
                    sum1 = sum1 / 10;
                    checksum = checksum + sum1;
                }

                else
                {
                    checksum = checksum + (2 * rem);
                }

                //Prog Checks
                //printf("Digit: %i\n", i);
                //printf("%lu\n", runsum);
                //printf("%i\n", checksum);
            }
        }

        //Check leading numbers
        rem = cc % 100000000000000;
        rem = cc - rem;
        rem = rem / 100000000000000;
        remv = rem / 10;
        //printf("%lu\n", rem);

        //Output for 16 digits
        if (checksum % 10 == 0)
        {
            if (rem == 51 || rem == 52 || rem == 53 || rem == 54 || rem == 55)
            {
                printf("MASTERCARD\n");
            }

            else if (remv == 4)
            {
                printf("VISA\n");
            }

            else
            {
                printf("INVALID\n");
            }
        }

        else
        {
            printf("INVALID\n");
        }
    }

    //Check if length = 15
    else if (cc > 99999999999999 && cc <= 999999999999999)
    {
        for (int i = 15; i > 0; i--)
        {
            //For odd digits
            if (i % 2 == 1)
            {
                rem = cc % div;
                rem = rem - runsum;
                runsum = runsum + rem;
                rem = rem / (div / 10);
                div = div * 10;
                checksum = checksum + rem;

                //Prog Checks
                //printf("Digit: %i\n", i);
                //printf("%lu\n", runsum);
                //printf("%i\n", checksum);
            }

            //For even digits
            else if (i % 2 == 0)
            {
                rem = cc % div;
                rem = rem - runsum;
                runsum = runsum + rem;
                rem = rem / (div / 10);
                div = div * 10;

                //Check if rem * 2 is 10 or greater
                if (rem * 2 > 9)
                {
                    rem = rem * 2;
                    sum = rem % 10;
                    checksum = checksum + sum;
                    sum1 = rem % 100;
                    sum1 = sum1 - sum;
                    sum1 = sum1 / 10;
                    checksum = checksum + sum1;
                }

                else
                {
                    checksum = checksum + (2 * rem);
                }

                //Prog Checks
                //printf("Digit: %i\n", i);
                //printf("%lu\n", runsum);
                //printf("%i\n", checksum);
            }
        }

        //Check leading numbers
        rem = cc % 10000000000000;
        rem = cc - rem;
        rem = rem / 10000000000000;
        remv = rem / 10;
        //printf("%lu\n", rem);

        //Output for 15 digits
        if (checksum % 10 == 0)
        {
            if (rem == 34 || rem == 37)
            {
                printf("AMEX\n");
            }

            else
            {
                printf("INVALID\n");
            }
        }

        else
        {
            printf("INVALID\n");
        }
    }

    //Check if length = 13
    else if (cc > 999999999999 && cc <= 9999999999999)
    {
        for (int i = 13; i > 0; i--)
        {
            //For odd digits
            if (i % 2 == 1)
            {
                rem = cc % div;
                rem = rem - runsum;
                runsum = runsum + rem;
                rem = rem / (div / 10);
                div = div * 10;
                checksum = checksum + rem;

                //Prog Checks
                //printf("Digit: %i\n", i);
                //printf("%lu\n", runsum);
                //printf("%i\n", checksum);
            }

            //For even digits
            if (i % 2 == 0)
            {
                rem = cc % div;
                rem = rem - runsum;
                runsum = runsum + rem;
                rem = rem / (div / 10);
                div = div * 10;

                //Check if rem * 2 is 10 or greater
                if (rem * 2 > 9)
                {
                    rem = rem * 2;
                    sum = rem % 10;
                    checksum = checksum + sum;
                    sum1 = rem % 100;
                    sum1 = sum1 - sum;
                    sum1 = sum1 / 10;
                    checksum = checksum + sum1;
                }

                else
                {
                    checksum = checksum + (2 * rem);
                }

                //Prog Checks
                //printf("Digit: %i\n", i);
                //printf("%lu\n", runsum);
                //printf("%i\n", checksum);
            }
        }

        //Check leading numbers
        rem = cc % 100000000000;
        rem = cc - rem;
        rem = rem / 100000000000;
        remv = rem / 10;
        //printf("%lu\n", rem);

        //Output for 13 digits
        if (checksum % 10 == 0)
        {
            if (remv == 4)
            {
                printf("VISA\n");
            }

            else
            {
                printf("INVALID\n");
            }
        }

        else
        {
            printf("INVALID\n");
        }
    }

    else
    {
        printf("INVALID\n");
    }

}