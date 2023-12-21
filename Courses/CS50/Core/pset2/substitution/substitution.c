#include <stdio.h>
#include <stdlib.h>
#include <cs50.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

//Check string function declaration
int checkstring(string c);
int checkrepeat(string c);

int main(int argc, string argv[])
{
    //Initial input check
    if (argc != 2 || checkstring(argv[1]) == 1)
    {
        printf("Usage: %s key\n", argv[0]);
        return 1;
    }
    
    //Pull key, normalize to lower case
    string cipher = argv[1];
    for (int i = 0; i < 26; i++)
    {
        //Change to all lowercase
        if (isupper(cipher[i]))
        {
            cipher[i] = tolower(cipher[i]);
        }
    }
    
    //Check repeat
    if (checkrepeat(cipher) == 1)
    {
        printf("Key must not contain repeated characters\n");
        return 1;
    }
    
    string text = get_string("plaintext: ");
    int len = strlen(text);
    char ciphertext [len];
    char c = '0';
    int p = 0;
    bool u = 0;

    for (int i = 0; i < strlen(text); i++)
    {
        
        //Pull char
        if (isalpha(text[i]))
        {
            c = text[i];

            //Check upper vs lower (flag)
            if (isupper(text[i]))
            {
                c = c + 32;
                u = 1;
            }
            
            c = c - 97;
            
            //Substitute
            p = c;
            c = cipher[p];
            
            //Re-upper case as req
            if (u == 1)
            {
                c = c - 32;
                u = 0;
            }
            
            //Place into ciphertext
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

//Function to check if cipher valid
int checkstring(string c)
{
    //Check length
    if (strlen(c) != 26)
    {
        return 1;
    }
    
    //Check if numbers
    for (int i = 0; i < strlen(c); i++)
    {
        if (isalpha(c[i]) == 0)
        {
            return 1;
        }
        
    }
    return 0;
}

//Function to check for repeats in cipher
int checkrepeat(string c)
{
    int p = 0;
    int check[26] = {0};
    for (int i = 0; i < strlen(c); i++)
    {
        p = c[i] - 97;
        
        //Check if letter already used
        if (check[p] == 1)
        {
            return 1;
        }
        
        //Set flag for letter usage
        else
        {
            check[p] = 1;
        }
    }
    return 0;
}