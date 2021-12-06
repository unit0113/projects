#include <stdio.h>
#include <stdlib.h>
#include <cs50.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

//Check string function declaration
int checkstring(string c);

int main(int argc, string argv[])
{
    //Check input
    if (argc != 2 || checkstring(argv[1]) == 1)
    {
        printf("Usage: %s key\n", argv[0]);
        return 1;
    }

    //Pull key
    int k = atoi(argv[1]);

    string text = get_string("plaintext: ");
    int len = strlen(text);
    char ciphertext [len];
    char c = '0';
    bool u = 0;

    for (int i = 0; i < strlen(text); i++)
    {
        //Check if letter
        if (isalpha(text[i]))
        {
            c = text[i];

            //Check upper vs lower (flag)
            if (isupper(text[i]))
            {
                c = c + 32;
                u = 1;
            }

            //Add cipher
            c = ((c + k - 97) % 26) + 97;

            //Convert back to upper if required
            if (u == 1)
            {
                c = c - 32;
                u = 0;
            }

            //Store to array
            ciphertext[i] = c;

        }

        //Non-letters
        else
        {
            ciphertext[i] = text[i];
        }

    }

    //Print result
    ciphertext[len] = '\0';
    printf("ciphertext: %s\n", ciphertext);
    return 0;
}

//Function to check if cipher is int
int checkstring(string c)
{
    for (int i = 0; i < strlen(c); i++)
    {
        if (isdigit(c[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}